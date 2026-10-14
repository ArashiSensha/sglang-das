"""Exercise the production agreement with asymmetric failures on two CPU ranks."""

import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"


def method(path, name, *, fused=True):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    ns = {
        "torch": torch,
        "envs": SimpleNamespace(
            SGLANG_MOONCAKE_FUSED_LOAD_AGREEMENT=SimpleNamespace(get=lambda: fused)
        ),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        agree = method(
            ROOT / "unified_cache/unified_cache_linker.py",
            "_build_and_agree_load_transfers",
        )
        # Success, one-rank allocation miss, one-rank exception, one-rank
        # stale cache, unexpected revalidation exception, legacy backend.
        for scenario in [
            "success",
            "allocation_miss",
            "build_error",
            "stale",
            "validation_error",
            "legacy",
        ]:
            reduced, cleaned, queried = [], [], []
            err = (
                RuntimeError("build")
                if scenario == "build_error" and rank == 1
                else None
            )
            constructed = not (
                rank == 1 and scenario in ["allocation_miss", "build_error"]
            )
            transfers = [("component", "allocated-slot")]

            def validate(ts):
                queried.append(ts)
                if scenario == "validation_error" and rank == 1:
                    raise RuntimeError("validation")
                return not (scenario == "stale" and rank == 1)

            def reduce(t, op):
                reduced.append(t.item())
                dist.all_reduce(t, op=op)

            def abort(req, ts, prefix, *, cancel_prefetch, pending):
                cleaned.append(ts)
                assert not cancel_prefetch
                if pending is not None:
                    raise pending

            fake = SimpleNamespace(
                _build_load_transfers=lambda *_: (transfers, constructed, err),
                cache_linker=SimpleNamespace(
                    local_load_revalidator=None if scenario == "legacy" else validate
                ),
                cache=SimpleNamespace(_all_reduce_attn_groups=reduce),
                _abort_disagreed_load=abort,
            )
            expected_error = rank == 1 and scenario in [
                "build_error",
                "validation_error",
            ]
            if expected_error:
                with pytest.raises(RuntimeError):
                    agree(fake, "request", ["key"], 64)
            else:
                ts, ok, validated = agree(fake, "request", ["key"], 64)
                assert ok == (scenario in ["success", "legacy"])
                assert validated == (scenario != "legacy")
            assert len(reduced) == 1
            assert len(cleaned) == (0 if scenario in ["success", "legacy"] else 1)
            assert len(queried) == int(constructed and scenario != "legacy")
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_asymmetric_failures(tmp_path):
    mp.spawn(worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


def test_enabled_fusion_uses_local_revalidation():
    hook = method(
        ROOT / "storage/mooncake_store/mooncake_direct_linker.py",
        "local_load_revalidator",
    )
    local = lambda _: True
    assert (
        hook(SimpleNamespace(rank0_lookup_group=None, _revalidate_load_local=local))
        is local
    )


def test_disabled_fusion_keeps_separate_revalidation():
    hook = method(
        ROOT / "storage/mooncake_store/mooncake_direct_linker.py",
        "local_load_revalidator",
        fused=False,
    )

    def unexpected(_):
        pytest.fail("disabled fusion must not validate during construction")

    validator = hook(
        SimpleNamespace(rank0_lookup_group=None, _revalidate_load_local=unexpected)
    )
    assert validator is None
    agree = method(
        ROOT / "unified_cache/unified_cache_linker.py",
        "_build_and_agree_load_transfers",
    )
    reductions = []
    fake = SimpleNamespace(
        _build_load_transfers=lambda *_: ([("component", "slot")], True, None),
        cache_linker=SimpleNamespace(local_load_revalidator=validator),
        cache=SimpleNamespace(
            _all_reduce_attn_groups=lambda t, op: reductions.append(t.item())
        ),
        _abort_disagreed_load=lambda *a, **kw: pytest.fail("unexpected abort"),
    )
    _, ok, revalidated = agree(fake, "request", ["key"], 64)
    assert ok and not revalidated
    assert reductions == [1]
