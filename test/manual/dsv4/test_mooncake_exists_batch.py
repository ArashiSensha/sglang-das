"""CPU-only semantics check of the actual storage method, without GPU imports."""

import ast
import dataclasses
import itertools
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS


@dataclasses.dataclass
class Result:
    kv_hit_pages: int
    extra_pool_hit_pages: dict
    restorable_prefix_pages: list


def method(path, batched=True, all_fn=None):
    tree = ast.parse(Path(path).read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeStore"
    )
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "batch_exists_v2"
    )
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    env = dict(
        envs=NS(SGLANG_MOONCAKE_BATCH_EXISTS=NS(get=lambda: batched)),
        PoolName=NS(KV="kv"),
        PoolHitPolicy=NS(ALL_PAGES="all", TRAILING_PAGES="trailing"),
        PoolTransferResult=Result,
    )
    if all_fn is not None:
        env["all"] = all_fn
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), env)
    return env["batch_exists_v2"]


class Store:

    def __init__(self, rows, physical=None):
        self.rows = rows
        self.mem_pool_host = NS(kv_buffer=None if physical is None else object())
        self.physical = physical
        self.calls = []
        self.constructed_pools = []

    def batch_exists(self, keys, extra):
        return self.physical

    def _get_hybrid_page_component_keys(self, keys, t):
        self.constructed_pools.append(t.name)
        return (
            [(t.name, i, j) for i in range(len(keys)) for j in range(t.multiplier)],
            t.multiplier,
        )

    def _tag_keys(self, keys):
        return [("model", k) for k in keys]

    def _batch_exist(self, keys):
        self.calls.append(keys)
        return [self.rows[name][i][j] for (tag, (name, i, j)) in keys]


def oracle(n, rows, transfers, physical=None):
    count = n if physical is None else physical
    prefixes = list(range(1, count + 1))
    hits = {"kv": count} if count else {}
    for t in transfers:
        if not prefixes:
            break
        allowed = []
        for k in range(1, count + 1):
            start = 0 if t.hit_policy == "all" else max(0, k - max(1, len(t.keys)))
            if all(
                rows[t.name][i] and all(x == 1 for x in rows[t.name][i])
                for i in range(start, k)
            ):
                allowed.append(k)
        if allowed:
            hits[t.name] = max(allowed)
        prefixes = [k for k in prefixes if k in allowed]
    return Result(max(prefixes, default=0), hits, prefixes)


def test_trailing_scan_short_circuits_without_window_copies():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
    )
    for batched in [False, True]:
        visited = 0

        def checked_all(values):
            nonlocal visited
            # A materialized window has already copied all its references
            # before all() can stop on its first miss.
            assert not isinstance(values, (list, tuple))

            def checked_values():
                nonlocal visited
                for value in values:
                    visited += 1
                    yield value

            return all(checked_values())

        fn = method(source, batched=batched, all_fn=checked_all)
        transfer = NS(
            name="swa", multiplier=1, keys=[None] * 256, hit_policy="trailing"
        )
        store = Store({"swa": [[0]] * 1024})
        assert fn(store, list(range(1024)), [transfer]).kv_hit_pages == 0
        # The disabled path also checks one component per logical page.
        assert visited == 1024 * (1 if batched else 2)


def test_disabled_lookup_stops_before_constructing_next_pool_keys():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
    )
    transfers = [
        NS(name=name, multiplier=1, keys=[None], hit_policy="all")
        for name in ["a", "b"]
    ]
    for batched in [False, True]:
        store = Store({"a": [[0]], "b": [[1]]})
        assert method(source, batched=batched)(store, [0], transfers).kv_hit_pages == 0
        assert len(store.calls) == 1
        # Batching intentionally constructs every pool before its single RPC.
        assert store.constructed_pools == (["a", "b"] if batched else ["a"])


def main():
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
    )
    candidate = method(sys.argv[1] if len(sys.argv) > 1 else source)
    unbatched = method(sys.argv[1] if len(sys.argv) > 1 else source, batched=False)
    baseline = method(sys.argv[2]) if len(sys.argv) > 2 else None
    rng = random.Random(42)
    cases = 0

    def check(n, rows, transfers, physical=None):
        nonlocal cases
        store = Store(rows, physical)
        got = candidate(store, list(range(n)), transfers)
        want = oracle(n, rows, transfers, physical)
        assert got == want, (n, rows, transfers, physical, got, want)
        assert unbatched(Store(rows, physical), list(range(n)), transfers) == want
        if baseline:
            assert got == baseline(Store(rows, physical), list(range(n)), transfers)
        assert len(store.calls) <= 1
        cases += 1

    for n in range(8):
        for mask in itertools.product([0, 1], repeat=n):
            for window in range(n + 2):
                for policy in ["all", "trailing"]:
                    t = NS(
                        name="swa",
                        multiplier=1,
                        keys=[None] * window,
                        hit_policy=policy,
                    )
                    check(n, {"swa": [[x] for x in mask]}, [t])
    for _ in range(5000):
        n = rng.randrange(40)
        rows = {}
        transfers = []
        for j in range(rng.randrange(8)):
            m = rng.randrange(1, 4)
            name = str(j)
            rows[name] = [
                [rng.choice([1, 1, 1, 1, 0, -1, -500]) for _ in range(m)]
                for _ in range(n)
            ]
            transfers.append(
                NS(
                    name=name,
                    multiplier=m,
                    keys=[None] * rng.randrange(n + 3),
                    hit_policy=rng.choice(["all", "trailing"]),
                )
            )
        check(n, rows, transfers, rng.choice([None, rng.randrange(n + 1)]))
    t = NS(name="swa", multiplier=1, keys=[None], hit_policy="trailing")
    a = candidate(Store({"swa": [[1], [0], [1]]}), [0, 1, 2], [t])
    b = candidate(Store({"swa": [[1], [1], [0]]}), [0, 1, 2], [t])
    assert set(a.restorable_prefix_pages) & set(b.restorable_prefix_pages) == {1}
    bad = Store({"swa": [[1]]})
    bad._batch_exist = lambda keys: []
    try:
        candidate(bad, [0], [t])
    except RuntimeError:
        pass
    else:
        raise AssertionError("short response was accepted")
    rows = {str(i): [[1] for _ in range(1024)] for i in range(6)}
    ts = [
        NS(
            name=str(i),
            multiplier=1,
            keys=[None],
            hit_policy="all" if i < 3 else "trailing",
        )
        for i in range(6)
    ]
    for label, fn in [("candidate", candidate), ("disabled", unbatched)] + (
        [("baseline", baseline)] if baseline else []
    ):
        s = Store(rows)
        start = time.perf_counter()
        for _ in range(30):
            r = fn(s, list(range(1024)), ts)
        assert r.kv_hit_pages == 1024
        assert len(s.calls) == 30 * (1 if label == "candidate" else 6)
        print(
            label,
            "mean_ms",
            (time.perf_counter() - start) * 1000 / 30,
            "RPCs_per_lookup",
            len(s.calls) / 30,
        )
    miss_rows = {"a": [[0]], "b": [[1]]}
    miss_transfers = [
        NS(name=name, multiplier=1, keys=[None], hit_policy="all") for name in miss_rows
    ]
    store = Store(miss_rows)
    assert unbatched(store, [0], miss_transfers).kv_hit_pages == 0
    assert len(store.calls) == 1  # Disabled path stops before the second pool RPC.

    # Require every PP shard for both full-prefix and trailing-window pools.
    class PPStore(Store):
        pp_rank = 0
        pp_size = 2

        def _get_hybrid_page_component_keys(self, keys, transfer):
            return ([f"{i}_pp0_{transfer.name}" for i in keys], 1)

        def _batch_exist(self, keys):
            self.calls.append(keys)
            return [int(key != "1_pp1_swa") for _, key in keys]

    pp_transfers = [
        NS(name="full", keys=[None], hit_policy="all"),
        NS(name="swa", keys=[None], hit_policy="trailing"),
    ]
    for fn in [candidate, unbatched] + ([baseline] if baseline else []):
        result = fn(PPStore({}), [0, 1, 2], pp_transfers, query_all_pp=True)
        assert result.restorable_prefix_pages == [1, 3]
        assert result.kv_hit_pages == 3
    for fn in [candidate, unbatched]:
        short = Store({"swa": [[1]]})
        short._batch_exist = lambda keys: []
        try:
            fn(short, [0], [t])
        except RuntimeError:
            pass
        else:
            raise AssertionError("short response was accepted")
    test_trailing_scan_short_circuits_without_window_copies()
    test_disabled_lookup_stops_before_constructing_next_pool_keys()
    print("EXISTS_SEMANTICS_PASSED_BOTH_MODES", cases)


if __name__ == "__main__":
    main()
