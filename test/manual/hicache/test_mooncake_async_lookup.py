"""CPU tests of the actual wrapper, including FIFO and cross-rank publication."""

import ast
import hashlib
import json
import logging
import threading
from array import array
from collections import deque, namedtuple
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py"
)


def wrapper_class():
    tree = ast.parse(SOURCE.read_text())
    names = {
        "ExternalCacheHitMarker",
        "_PendingOffload",
        "_PendingLookup",
        "UnifiedCacheLinkerWrapper",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    ns = dict(
        torch=torch,
        hashlib=hashlib,
        json=json,
        array=array,
        deque=deque,
        NamedTuple=NamedTuple,
        Future=Future,
        ThreadPoolExecutor=ThreadPoolExecutor,
        logger=logging.getLogger(__name__),
        PoolName=NS(KV="kv", SWA="swa", MAMBA="mamba"),
        LinkerTransferPhase=NS(LOOKUP="lookup"),
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    return ns["UnifiedCacheLinkerWrapper"]


class Key:
    def __init__(self, tokens, salt=None):
        self.token_ids = tokens
        self.extra_key = None
        self.cache_salt = salt
        self.is_bigram = False

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, selection):
        return Key(self.token_ids[selection], self.cache_salt)

    def raw_token_ids(self):
        return self.token_ids


Result = namedtuple(
    "Result",
    "device_indices best_match_node last_host_node host_hit_length swa_host_hit_length mamba_host_hit_length",
)


def result(device=0):
    return Result(torch.arange(device), 0, None, 0, 0, 0)


class Backend:
    async_lookup_enabled = True

    def __init__(self, probe=lambda rid, transfers: [1, 3]):
        self.probe = probe
        self.calls = []
        self.closed = False

    def lookup_in_worker(self, rid, transfers):
        assert threading.current_thread() is not threading.main_thread()
        assert not self.closed
        self.calls.append(rid)
        return self.probe(rid, transfers)

    def cancel_host_prefetch(self, rid):
        pass

    def cancel_queued_load(self, rid):
        return False

    def reset(self):
        pass

    def close(self):
        self.closed = True


def make_wrapper(backend=None, reducer=None, workers=1):
    backend = backend or Backend()
    backend.async_lookup_workers = workers
    cache = NS(
        page_size=1,
        pp_size=1,
        tree_core=NS(),
        _components_tuple=[
            NS(
                build_external_linker_transfer=lambda phase, node, hashes: NS(
                    name="kv", keys=hashes
                )
            )
        ],
        _all_reduce_attn_groups=reducer or (lambda *args: None),
    )
    wrapper = wrapper_class()(cache, backend)
    wrapper._tail_hashes = lambda key, result, device_hit_len: [
        str(t) for t in key.token_ids[device_hit_len:]
    ]
    return wrapper, backend


def finish(wrapper):
    wrapper.drain_lookups(wrapper.wait_pending_lookups())


def test_nonblocking_dedup_and_published_results():
    entered, release = threading.Event(), threading.Event()

    def probe(rid, transfers):
        entered.set()
        assert release.wait(5)
        return [1, 3]

    wrapper, backend = make_wrapper(Backend(probe))
    req, key = NS(rid="a"), Key([10, 20, 30])
    try:
        assert wrapper.match(key, req, result()).host_hit_length == 0
        assert entered.wait(5)
        assert wrapper.has_pending_lookup("a")
        assert wrapper.match(key, req, result()).host_hit_length == 0
        assert len(wrapper.lookup_queue) == 1
        assert not wrapper.lookup_queue[0].future.done()
        release.set()
        finish(wrapper)
        assert not wrapper.has_pending_lookup("a")
        assert wrapper.match(key, req, result()).host_hit_length == 3
        assert wrapper.match(key, req, result()).host_hit_length == 3
        assert backend.calls == ["a"]
    finally:
        release.set()
        wrapper.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_cancelled_rid_and_changed_device_prefix(workers):
    wrapper, backend = make_wrapper(workers=workers)
    req, key = NS(rid="a"), Key([10, 20, 30])
    try:
        wrapper.match(key, req, result())
        old = wrapper.lookup_queue[0]
        wrapper.release_request("a")
        wrapper.match(key, req, result())
        old.future.result(timeout=5)
        wrapper.drain_lookups(1)
        assert wrapper.has_pending_lookup("a")
        assert not wrapper.lookup_results
        finish(wrapper)
        assert wrapper.match(key, req, result()).host_hit_length == 3
        # Reprobe when the L1 anchor moves; never apply stale relative offsets.
        assert wrapper.match(key, req, result(1)).host_hit_length == 0
        assert wrapper.has_pending_lookup("a")
        finish(wrapper)
        assert wrapper.match(key, req, result(1)).host_hit_length == 1
        # A fully resident prefix can proceed without waiting for another probe.
        assert wrapper.match(key, req, result(3)).host_hit_length == 0
        assert not wrapper.has_pending_lookup("a")
    finally:
        wrapper.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_miss_error_reset_and_query_identity(workers):
    def failed(rid, transfers):
        raise ValueError("injected query error")

    wrapper, backend = make_wrapper(Backend(failed), workers=workers)
    req, key = NS(rid="a"), Key([1, 2, 3])
    try:
        wrapper.match(key, req, result())
        finish(wrapper)
        for _ in range(3):
            assert wrapper.match(key, req, result()).host_hit_length == 0
        assert len(backend.calls) == 1
        # Same rid and length but different tokens must not reuse the miss.
        wrapper.match(Key([7, 8, 9]), req, result())
        assert wrapper.has_pending_lookup("a")
        wrapper.reset()
        assert not wrapper.lookup_queue and not wrapper.lookup_results
        wrapper.match(key, req, result())
        finish(wrapper)
        assert len(backend.calls) == 3
    finally:
        wrapper.close()


@pytest.mark.parametrize("first_fails", [False, True])
def test_waits_for_entire_cohort_before_publication(first_fails):
    wrapper, _ = make_wrapper()
    reached_second = threading.Event()
    release_second = threading.Event()
    errors = []

    class ObservedFuture(Future):
        def result(self, timeout=None):
            reached_second.set()
            assert release_second.wait(5)
            return super().result(timeout=5)

    try:
        key = Key([10, 20, 30])
        for rid in ("first", "second"):
            wrapper.match(key, NS(rid=rid), result())
        # Substitute controlled futures after the real probes finish.
        for pending in wrapper.lookup_queue:
            pending.future.result(timeout=5)
        first, second = Future(), ObservedFuture()
        if first_fails:
            first.set_exception(ValueError("injected query error"))
        else:
            first.set_result([1])
        for index, future in enumerate((first, second)):
            pending = wrapper.lookup_queue[index]._replace(future=future)
            wrapper.lookup_queue[index] = pending
            wrapper.pending_lookups[pending.rid] = pending

        def complete_second():
            try:
                assert reached_second.wait(5)
                assert not wrapper.lookup_results
                assert len(wrapper.lookup_queue) == 2
                second.set_result([3])
            except BaseException as error:
                errors.append(error)
            finally:
                release_second.set()

        controller = threading.Thread(target=complete_second)
        controller.start()
        try:
            finish(wrapper)
        finally:
            release_second.set()
            controller.join(timeout=5)
        assert not controller.is_alive()
        assert not errors
        assert wrapper.match(key, NS(rid="first"), result()).host_hit_length == (
            0 if first_fails else 1
        )
        assert wrapper.match(key, NS(rid="second"), result()).host_hit_length == 3
        assert not wrapper.lookup_queue
    finally:
        release_second.set()
        wrapper.close()


def test_wait_uses_fixed_snapshot_and_empty_queue_is_noop():
    wrapper, _ = make_wrapper()
    later = NS(future=Future())

    class AppendFuture(Future):
        def result(self, timeout=None):
            wrapper.lookup_queue.append(later)
            return super().result(timeout=5)

    try:
        assert wrapper.wait_pending_lookups() == 0
        first = AppendFuture()
        first.set_result([])
        wrapper.lookup_queue.append(NS(future=first))
        assert wrapper.wait_pending_lookups() == 1
        assert len(wrapper.lookup_queue) == 2
        assert not later.future.done()
    finally:
        wrapper.lookup_queue.clear()
        wrapper.close()


def _rank_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )

    def reduce(mask, op):
        assert threading.current_thread() is threading.main_thread()
        dist.all_reduce(mask, op=op)

    backend = Backend(
        lambda rid, transfers: (
            ([1, 3] if rank == 0 else [2, 3])
            if rid == "hit"
            else ([1] if rank == 0 else [])
        )
    )
    wrapper, _ = make_wrapper(backend, reduce, workers=2)
    try:
        key = Key([10, 20, 30])
        for rid in ("hit", "miss"):
            wrapper.match(key, NS(rid=rid), result())
        # Wait for the same fixed cohort on every rank, then publish both
        # results in one scheduler-thread collective.
        count = torch.tensor([wrapper.wait_pending_lookups()], dtype=torch.int)
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
        assert count.item() == 2
        wrapper.drain_lookups(int(count.item()))
        assert wrapper.match(key, NS(rid="hit"), result()).host_hit_length == 3
        assert not wrapper.has_pending_lookup("miss")
        assert wrapper.match(key, NS(rid="miss"), result()).host_hit_length == 0
    finally:
        wrapper.close()
        dist.destroy_process_group()


def test_cross_rank_sparse_intersection(tmp_path):
    mp.spawn(_rank_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


def test_synchronous_default_is_preserved():
    backend = Backend()
    backend.async_lookup_enabled = False
    backend.lookup = lambda rid, transfers: [1, 3]
    wrapper, _ = make_wrapper(backend)
    try:
        assert wrapper.lookup_executor is None
        assert (
            wrapper.match(Key([1, 2, 3]), NS(rid="sync"), result()).host_hit_length == 3
        )
        assert not wrapper.lookup_queue
    finally:
        wrapper.close()


def test_close_joins_worker_before_storage_close():
    entered, release = threading.Event(), threading.Event()

    def probe(rid, transfers):
        entered.set()
        assert release.wait(5)
        assert not backend.closed
        return []

    backend = Backend(probe)
    wrapper, _ = make_wrapper(backend)
    wrapper.match(Key([1]), NS(rid="pending"), result())
    assert entered.wait(5)
    closer = threading.Thread(target=wrapper.close)
    closer.start()
    try:
        assert not backend.closed
    finally:
        release.set()
        closer.join(timeout=5)
    assert not closer.is_alive() and backend.closed
    assert not wrapper.lookup_queue


def test_lookup_worker_uses_local_metadata_without_collectives():
    source = SOURCE.parents[1] / "storage/mooncake_store/mooncake_direct_linker.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeDirectLinker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "lookup_in_worker"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = {"PoolName": NS(KV="kv")}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    calls = []
    backend = NS(
        _lookup_local=lambda *args: calls.append(args) or [1, 3],
    )
    transfers = [NS(name="kv", keys=["a", "b", "c"])]
    worker = namespace["lookup_in_worker"]
    assert worker(backend, "r", transfers) == [1, 3]
    assert len(calls) == 1


def test_two_workers_complete_out_of_order_but_publish_fifo():
    first_entered, release_first = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    reductions = []

    def probe(rid, transfers):
        if rid == "first":
            first_entered.set()
            assert release_first.wait(5)
            return [1]
        assert first_entered.wait(5)
        return [2]

    def reduce(mask, op):
        assert threading.get_ident() == main_thread
        reductions.append(mask.tolist())

    wrapper, _ = make_wrapper(Backend(probe), reduce, workers=2)
    key = Key([10, 20, 30])
    try:
        wrapper.match(key, NS(rid="first"), result())
        wrapper.match(key, NS(rid="second"), result())
        # The second probe must finish while the first is still blocked.
        assert wrapper.lookup_queue[1].future.result(timeout=5) == [2]
        assert not wrapper.lookup_queue[0].future.done()
        wrapper.drain_lookups(0)
        assert not reductions
        release_first.set()
        finish(wrapper)
        assert len(reductions) == 1
        assert wrapper.match(key, NS(rid="first"), result()).host_hit_length == 1
        assert wrapper.match(key, NS(rid="second"), result()).host_hit_length == 2
    finally:
        release_first.set()
        wrapper.close()


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_two_workers_join_before_backend_cleanup(operation):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    exited = [threading.Event(), threading.Event()]
    cleanup_started, cleanup_finished = threading.Event(), threading.Event()
    errors = []

    def probe(rid, transfers):
        index = int(rid)
        entered[index].set()
        assert release[index].wait(5)
        assert not backend.closed
        exited[index].set()
        return [1]

    backend = Backend(probe)

    def reset():
        assert all(event.is_set() for event in exited)

    def close():
        assert all(event.is_set() for event in exited)
        backend.closed = True

    backend.reset = reset
    backend.close = close
    wrapper, _ = make_wrapper(backend, workers=2)
    cleaner = None
    try:
        for index in range(2):
            wrapper.match(Key([1]), NS(rid=str(index)), result())
        assert all(event.wait(5) for event in entered)
        old_executor = wrapper.lookup_executor

        def cleanup():
            cleanup_started.set()
            try:
                getattr(wrapper, operation)()
            except BaseException as error:
                errors.append(error)
            finally:
                cleanup_finished.set()

        cleaner = threading.Thread(target=cleanup)
        cleaner.start()
        assert cleanup_started.wait(5)
        release[0].set()
        assert exited[0].wait(5)
        assert not cleanup_finished.is_set()
        assert not backend.closed
        release[1].set()
        cleaner.join(timeout=5)
        assert not cleaner.is_alive()
        assert not errors
        assert not wrapper.lookup_queue and not wrapper.pending_lookups
        if operation == "reset":
            assert wrapper.lookup_executor is not old_executor
            # Verify the replacement pool still runs two probes concurrently.
            for event in entered + release + exited:
                event.clear()
            for index in range(2):
                wrapper.match(Key([1]), NS(rid=str(index)), result())
            assert all(event.wait(5) for event in entered)
        else:
            assert wrapper.lookup_executor is None and backend.closed
    finally:
        for event in release:
            event.set()
        if cleaner is not None:
            cleaner.join(timeout=5)
        if not backend.closed:
            wrapper.close()


@pytest.mark.parametrize(
    "enabled,workers,error",
    [
        (True, 1, False),
        (True, 2, False),
        (True, 0, True),
        (True, -1, True),
        (False, 0, False),
    ],
)
def test_backend_worker_configuration(enabled, workers, error):
    # Execute the real configuration block without constructing GPU pools.
    source = SOURCE.parents[1] / "storage/mooncake_store/mooncake_direct_linker.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeDirectLinker"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )

    def assigns(node, attribute):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == attribute for t in node.targets
        )

    start = next(
        i for i, node in enumerate(init.body) if assigns(node, "async_lookup_enabled")
    )
    end = next(
        i for i, node in enumerate(init.body) if assigns(node, "read_layout_enabled")
    )
    module = ast.Module(body=init.body[start:end], type_ignores=[])
    backend = NS()
    ns = dict(
        self=backend,
        params=NS(pp_size=1),
        logger=logging.getLogger(__name__),
        envs=NS(
            SGLANG_MOONCAKE_ASYNC_LOOKUP=NS(get=lambda: enabled),
            SGLANG_MOONCAKE_ASYNC_LOOKUP_WORKERS=NS(get=lambda: workers),
        ),
    )
    code = compile(ast.fix_missing_locations(module), str(source), "exec")
    if error:
        with pytest.raises(ValueError, match="ASYNC_LOOKUP_WORKERS must be >= 1"):
            exec(code, ns)
    else:
        exec(code, ns)
        assert backend.async_lookup_workers == (workers if enabled else 1)


def extracted_methods(relative_path, class_name, names, namespace):
    source = SOURCE.parents[5] / relative_path
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    cls.bases = []
    cls.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[class_name]


def test_current_cohort_submits_all_before_waiting():
    # Both probes must enter before either is allowed to complete. Waiting
    # inside the submission loop would time out, instead of querying in parallel.
    barrier = threading.Barrier(2, timeout=5)
    main_thread = threading.get_ident()
    reductions = []

    def probe(rid, transfers):
        barrier.wait()
        return [3]

    def reduce(mask, op):
        assert threading.get_ident() == main_thread
        reductions.append(mask.tolist())

    wrapper, _ = make_wrapper(Backend(probe), reduce, workers=2)
    cache_type = extracted_methods(
        "python/sglang/srt/mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        {"submit_external_lookups"},
        {"envs": NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False))},
    )
    cache = cache_type()
    cache.linker, cache.disable, cache.pp_size = wrapper, False, 1
    cache.match_prefix = lambda params: wrapper.match(params.key, params.req, result())
    cache.check_hicache_events = lambda: finish(wrapper)
    key = Key([10, 20, 30])
    reqs = [NS(rid=rid) for rid in ("a", "b")]
    for req in reqs:
        req.prepare_external_lookup = lambda cache, req=req: NS(key=key, req=req)
    try:
        cache.submit_external_lookups(reqs)
        cache.check_hicache_events()
        assert len(reductions) == 1
        assert not wrapper.pending_lookups
        # Admission can consume both hits in this same call/round.
        for req in reqs:
            assert wrapper.match(key, req, result()).host_hit_length == 3
        assert sorted(wrapper.cache_linker.calls) == ["a", "b"]
    finally:
        wrapper.close()


@pytest.mark.parametrize(
    "logprob_cap,hidden_cap,swa_tail,overrides,expected",
    [
        (None, None, 0, False, 9),
        (4, None, 0, False, 4),
        (None, 5, 0, False, 5),
        (None, None, 3, False, 7),
        (4, 5, 3, False, 4),
        (None, None, 0, True, None),
    ],
)
def test_probe_and_admission_share_prefix_limits(
    logprob_cap, hidden_cap, swa_tail, overrides, expected
):
    from array import array

    req_type = extracted_methods(
        "python/sglang/srt/managers/schedule_batch.py",
        "Req",
        {
            "prepare_external_lookup",
            "_prefix_match_params",
            "_refresh_fill_ids",
            "_compute_max_prefix_len",
        },
        {"array": array, "RadixKey": NS, "MatchPrefixParams": NS},
    )
    req = req_type()
    req.external_cache_hit_length = None
    req.session = None
    req.is_dllm = lambda: False
    req.origin_input_ids = array("q", range(8))
    req.output_ids = array("q", [8, 9])
    req.full_untruncated_fill_ids = array("q")
    req.return_logprob = logprob_cap is not None
    req.logprob_start_len = logprob_cap if logprob_cap is not None else -1
    req.pd_hidden_max_prefix_len = hidden_cap
    req.positional_embed_overrides = object() if overrides else None
    req.extra_key, req.cache_salt = "adapter", "salt"
    cache = NS(swa_reprefill_tail_tokens=lambda: swa_tail)
    probe = req.prepare_external_lookup(cache)
    req._refresh_fill_ids()
    admission = req._prefix_match_params(cache, cow_mamba=True)
    assert probe.key == admission.key
    assert probe.key.limit == expected
    assert probe.key.extra_key == "adapter" and probe.key.cache_salt == "salt"
    assert not probe.cow_mamba and admission.cow_mamba
    assert probe.return_full_match == bool(swa_tail)
    assert len(probe.key.token_ids) == (0 if overrides else 10)
    # Special input/state paths must not be initialized by the pre-probe.
    req._refresh_fill_ids = lambda: pytest.fail("unexpected initialization")
    req.session = object()
    assert req.prepare_external_lookup(cache) is None
    req.external_cache_hit_length = None
    req.session = None
    req.is_dllm = lambda: True
    assert req.prepare_external_lookup(cache) is None


@pytest.mark.parametrize("dfs", [False, True])
def test_scheduler_prepares_cohort_before_admission_in_same_call(dfs):
    events = []

    class Admitted(Exception):
        pass

    class Adder:
        def __init__(self, *args, **kwargs):
            self.can_run_list = []

        def add_one_req(self, req, **kwargs):
            expected = (
                ["prepare", "poll", "dfs", "states", "init"]
                if dfs
                else ["prepare", "poll", "init"]
            )
            assert events == expected
            raise Admitted

    scheduler_type = extracted_methods(
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler",
        {"_get_new_batch_prefill_raw"},
        {
            "get_memory": lambda: NS(enable_flexkv=False),
            "TEST_RETRACT": False,
            "get_schedule": lambda: NS(prefill_max_requests=16),
            "PrefillAdder": Adder,
            "DisaggregationMode": NS(PREFILL="prefill"),
        },
    )
    scheduler = scheduler_type()
    req = NS(rid="a", init_next_round_input=lambda cache: events.append("init"))
    scheduler.waiting_queue = [req, NS(rid="b")]

    def prepare(requests):
        assert [r.rid for r in requests] == ["a", "b"]
        events.append("prepare")

    scheduler.tree_cache = NS(
        linker=NS(async_lookup_enabled=True),
        check_hicache_events=lambda: events.append("poll"),
        submit_external_lookups=prepare,
        is_external_lookup_pending=lambda rid: False,
    )
    scheduler.grammar_manager = NS(has_waiting_grammars=lambda: False)
    scheduler.enable_waiting_queue_dfs_prefetch = dfs
    scheduler._complete_waiting_queue_prefetch_lookups = lambda: (
        events.append("dfs") or []
    )

    def states(rids, *, pending_lookup_rids):
        assert events == ["prepare", "poll", "dfs"]
        assert pending_lookup_rids == []
        events.append("states")
        return {rid: "not_tracked" for rid in rids}

    scheduler.tree_cache.get_waiting_queue_prefetch_admission_states = states
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_unified_cache_external_linker = True
    scheduler.enable_priority_preemption = scheduler.is_hybrid_swa = False
    scheduler.chunked_req = scheduler.min_free_slots_delayer = None
    scheduler.get_num_allocatable_reqs = lambda running_bs: 2
    scheduler.policy = NS(calc_priority=lambda *args: None)
    scheduler.chunked_prefill_size = 512
    scheduler.tp_worker = NS(model_runner=NS(attn_backend=NS()))
    scheduler.page_size = 1
    scheduler.token_to_kv_pool_allocator = None
    scheduler.new_token_ratio_tracker = NS(current=1)
    scheduler.max_prefill_tokens = 1024
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = scheduler.max_running_requests = 16
    scheduler.dllm_config = None
    scheduler.enable_lora = scheduler.enable_hicache_storage = False
    scheduler.req_to_token_pool = NS(available_size=lambda: 2)
    scheduler.disaggregation_mode = "prefill"
    scheduler.truncation_align_size = None
    with pytest.raises(Admitted):
        scheduler._get_new_batch_prefill_raw(None, NS(batch_is_full=False, reqs=[]))


def prefetch_scheduler(wrapper):
    import queue

    cls = extracted_methods(
        "python/sglang/srt/managers/scheduler.py",
        "Scheduler",
        {
            "_prefetch_kvcache",
            "_enqueue_waiting_queue_prefetch",
            "_complete_waiting_queue_prefetch_lookups",
        },
        {"DAS_PREFETCH_TRACE_ENABLED": False},
    )
    scheduler = cls()
    scheduler.enable_waiting_queue_dfs_prefetch = True
    scheduler.enable_hicache_storage = False
    scheduler._bg_prefetch_rooms = None
    scheduler._bg_prefetch_jobs = queue.Queue()
    scheduler.waiting_queue = []
    scheduler.disagg_prefill_bootstrap_queue = NS(queue=[])
    scheduler.tree_cache = NS(
        linker=wrapper,
        is_external_lookup_pending=wrapper.has_pending_lookup,
        # Observe the real lookup wrapper's published/rematched hit marker at
        # the point the scheduler freezes the prefetch snapshot.
        prepare_external_linker_prefetch=lambda req: NS(
            rid=req.rid, locally_eligible=req.rid in wrapper.hit_markers
        ),
    )
    return scheduler


def prefetch_request(wrapper, rid="a"):
    req = NS(rid=rid, prefill_attempt_count=0, finished=lambda: False)
    req.key = Key([10, 20, 30])
    req.device_hit = 0

    def init(cache, cow_mamba=False):
        matched = wrapper.match(req.key, req, result(req.device_hit))
        req.host_hit_length = matched.host_hit_length

    req.init_next_round_input = init
    return req


@pytest.mark.parametrize("hit", [True, False])
@pytest.mark.parametrize("bootstrap", [True, False])
def test_async_lookup_defers_dfs_snapshot_until_published(hit, bootstrap):
    wrapper, backend = make_wrapper(Backend(lambda *_: [3] if hit else []))
    scheduler = prefetch_scheduler(wrapper)
    req = prefetch_request(wrapper)
    target = (
        scheduler.disagg_prefill_bootstrap_queue.queue
        if bootstrap
        else scheduler.waiting_queue
    )
    target.append(req)
    try:
        scheduler._prefetch_kvcache(req)
        wrapper.wait_pending_lookups()  # RPC done does not mean rank-wide publication.
        assert scheduler._complete_waiting_queue_prefetch_lookups() == [req.rid]
        assert scheduler._bg_prefetch_jobs.empty()
        assert req.waiting_queue_dfs_lookup_pending
        finish(wrapper)
        scheduler._complete_waiting_queue_prefetch_lookups()
        job = scheduler._bg_prefetch_jobs.get_nowait()
        assert job.rid == req.rid
        assert job.locally_eligible is hit
        assert not req.waiting_queue_dfs_lookup_pending
        scheduler._complete_waiting_queue_prefetch_lookups()
        assert scheduler._bg_prefetch_jobs.empty()
        assert backend.calls == ["a"]
    finally:
        wrapper.lookup_executor.shutdown()


def test_async_dfs_rematch_waits_for_replacement_lookup():
    wrapper, backend = make_wrapper()
    scheduler = prefetch_scheduler(wrapper)
    req = prefetch_request(wrapper)
    scheduler.waiting_queue = [req]
    try:
        scheduler._prefetch_kvcache(req)
        finish(wrapper)
        req.key = Key([40, 50, 60])  # Prefix changed before the prefetch snapshot.
        scheduler._complete_waiting_queue_prefetch_lookups()
        assert scheduler._bg_prefetch_jobs.empty()
        assert req.waiting_queue_dfs_lookup_pending
        finish(wrapper)
        scheduler._complete_waiting_queue_prefetch_lookups()
        assert scheduler._bg_prefetch_jobs.get_nowait().locally_eligible
        assert backend.calls == ["a", "a"]
    finally:
        wrapper.lookup_executor.shutdown()


@pytest.mark.parametrize("state", ["finished", "removed", "retried"])
def test_async_dfs_does_not_resubmit_cancelled_or_retried_request(state):
    wrapper, _ = make_wrapper()
    scheduler = prefetch_scheduler(wrapper)
    req = prefetch_request(wrapper)
    scheduler.waiting_queue = [req]
    try:
        scheduler._prefetch_kvcache(req)
        finish(wrapper)
        if state == "finished":
            req.finished = lambda: True
        elif state == "removed":
            scheduler.waiting_queue.clear()
        else:
            req.prefill_attempt_count = 1
        scheduler._complete_waiting_queue_prefetch_lookups()
        assert scheduler._bg_prefetch_jobs.empty()
    finally:
        wrapper.lookup_executor.shutdown()


def test_async_dfs_l1_hit_and_queue_move_are_submitted_once():
    wrapper, backend = make_wrapper()
    scheduler = prefetch_scheduler(wrapper)
    req = prefetch_request(wrapper)
    req.device_hit = 3
    scheduler.disagg_prefill_bootstrap_queue.queue = [req]
    try:
        scheduler._prefetch_kvcache(req)
        assert scheduler._bg_prefetch_jobs.empty()
        scheduler.waiting_queue = [req]  # Defend against a transient duplicate.
        scheduler._complete_waiting_queue_prefetch_lookups()
        assert not scheduler._bg_prefetch_jobs.get_nowait().locally_eligible
        assert scheduler._bg_prefetch_jobs.empty()
        assert backend.calls == []
    finally:
        wrapper.lookup_executor.shutdown()


def test_sync_dfs_prefetch_still_submits_at_ingress():
    wrapper, backend = make_wrapper()
    wrapper.async_lookup_enabled = False
    backend.lookup = lambda *_: [3]
    scheduler = prefetch_scheduler(wrapper)
    req = prefetch_request(wrapper)
    try:
        scheduler._prefetch_kvcache(req)
        assert scheduler._bg_prefetch_jobs.get_nowait().locally_eligible
        assert not getattr(req, "waiting_queue_dfs_lookup_pending", False)
    finally:
        wrapper.lookup_executor.shutdown()


def _dfs_lookup_rank_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    cache_type = extracted_methods(
        "python/sglang/srt/mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        {"get_waiting_queue_prefetch_admission_states"},
        {"torch": torch},
    )
    cache = cache_type()
    cache.attn_cp_group = dist.group.WORLD
    cache.attn_tp_group = None
    cache.tp_world_size = 2
    cache.tp_group = dist.group.WORLD
    cache._all_reduce_attn_groups = lambda tensor, op: dist.all_reduce(tensor, op=op)
    # Rank 0 has not submitted a job yet; rank 1 already completed its prefetch.
    cache.linker = NS(
        get_host_prefetch_admission_state=lambda rid: (
            "not_tracked" if rank == 0 else "dfs_prefetched"
        )
    )
    try:
        states = cache.get_waiting_queue_prefetch_admission_states(
            [] if rank == 0 else ["a"],
            pending_lookup_rids=["a"] if rank == 0 else [],
        )
        assert states["a"] == "pending"
        # Once lookup finishes, a genuine non-hit takes the usual agreed
        # fallback, rather than one rank loading while its peer is waiting.
        states = cache.get_waiting_queue_prefetch_admission_states(["a"])
        assert states["a"] == "terminal"
        cache.linker.get_host_prefetch_admission_state = lambda rid: "dfs_prefetched"
        states = cache.get_waiting_queue_prefetch_admission_states(["a"])
        assert states["a"] == "dfs_prefetched"
    finally:
        dist.destroy_process_group()


def test_dfs_lookup_pending_is_agreed_across_unequal_rank_queues(tmp_path):
    mp.spawn(
        _dfs_lookup_rank_worker,
        args=((tmp_path / "dfs-gloo").as_uri(),),
        nprocs=2,
        join=True,
    )


class ControlledLookupExecutor:
    """Keep probes pending until the test explicitly completes them."""

    def __init__(self):
        self.futures = []

    def submit(self, *args):
        future = Future()
        self.futures.append(future)
        return future

    def shutdown(self, wait=True):
        for future in self.futures:
            future.cancel()


def ready_lookup_cache(reducer=None, cp_group=None, tp_group=None, world_size=1):
    backend = Backend()
    backend.waiting_queue_prefetch_enabled = lambda: True
    wrapper, _ = make_wrapper(backend, reducer)
    wrapper.lookup_executor.shutdown()
    wrapper.lookup_executor = ControlledLookupExecutor()
    cache_type = extracted_methods(
        "python/sglang/srt/mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        {"_publish_ready_external_lookups", "check_hicache_events"},
        {"torch": torch},
    )
    cache = cache_type()
    cache.linker = wrapper
    cache.attn_cp_group = cp_group
    cache.attn_tp_group = tp_group
    cache.tp_world_size = world_size
    cache.tp_group = dist.group.WORLD if world_size > 1 else None
    cache._all_reduce_attn_groups = reducer or (lambda *args: None)
    return cache, wrapper


def controlled_probe(wrapper, rid, tokens=(10, 20, 30), device=0, salt=None):
    wrapper.match(Key(list(tokens), salt), NS(rid=rid), result(device))
    return wrapper.pending_lookups[rid].future


def test_dfs_ready_probe_starts_prefetch_without_waiting_for_slow_probe():
    cache, wrapper = ready_lookup_cache()
    scheduler = prefetch_scheduler(wrapper)
    slow = prefetch_request(wrapper, "slow")
    fast = prefetch_request(wrapper, "fast")
    scheduler.waiting_queue = [slow, fast]
    try:
        for req in scheduler.waiting_queue:
            scheduler._prefetch_kvcache(req)
        wrapper.pending_lookups["fast"].future.set_result([1, 3])
        cache._publish_ready_external_lookups()
        assert scheduler._complete_waiting_queue_prefetch_lookups() == ["slow"]
        job = scheduler._bg_prefetch_jobs.get_nowait()
        assert job.rid == "fast" and job.locally_eligible
        assert scheduler._bg_prefetch_jobs.empty()
        assert not wrapper.pending_lookups["slow"].future.done()
        wrapper.pending_lookups["slow"].future.set_result([1])
        cache._publish_ready_external_lookups()
        assert scheduler._complete_waiting_queue_prefetch_lookups() == []
        assert scheduler._bg_prefetch_jobs.get_nowait().rid == "slow"
    finally:
        wrapper.close()


def test_dfs_ready_publication_preserves_completed_io_progress():
    cache, wrapper = ready_lookup_cache()
    events = []
    try:
        controlled_probe(wrapper, "slow")
        wrapper.num_completed_loads = lambda: 1
        wrapper.num_completed_offloads = lambda: 1
        wrapper.wait_pending_lookups = lambda: pytest.fail(
            "must not wait for DFS lookup"
        )
        cache._collect_failed_linker_loads = lambda n: events.append(("load", n))
        wrapper.take_completed_offloads = lambda n: [True] * n
        wrapper.commit_completed_offloads = lambda r: events.append(("offload", r))
        cache.check_hicache_events()
        assert events == [("load", 1), ("offload", [True])]
        assert wrapper.has_pending_lookup("slow")
    finally:
        wrapper.close()


def test_dfs_ready_publication_rejects_stale_and_reused_rid_results():
    cache, wrapper = ready_lookup_cache()
    try:
        old = controlled_probe(wrapper, "reused")
        wrapper.release_request("reused")
        new = controlled_probe(wrapper, "reused")
        old.set_result([3])
        cache._publish_ready_external_lookups()
        assert not wrapper.lookup_results
        assert wrapper.pending_lookups["reused"].future is new
        assert len(wrapper.lookup_queue) == 1
        new.set_exception(RuntimeError("lookup failed"))
        cache._publish_ready_external_lookups()
        assert wrapper.lookup_results["reused"][1] == 0
        changed = controlled_probe(wrapper, "reused", tokens=(11, 20, 30))
        changed.set_result([1])
        cache._publish_ready_external_lookups()
        assert wrapper.lookup_results["reused"][1] == 1
    finally:
        wrapper.close()


def test_ready_lookup_mode_requires_async_and_resolved_dfs():
    cache, wrapper = ready_lookup_cache()
    try:
        assert wrapper.ready_lookup_publication
        wrapper.cache_linker.waiting_queue_prefetch_enabled = lambda: False
        assert not wrapper.ready_lookup_publication
        wrapper.cache_linker.waiting_queue_prefetch_enabled = lambda: True
        wrapper.async_lookup_enabled = False
        assert not wrapper.ready_lookup_publication
    finally:
        wrapper.close()


def _ready_lookup_rank_worker(rank, rendezvous, topology):
    size = 4 if topology == "cp_tp" else 2
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=30),
    )
    cp_group, tp_group = None, None
    if topology == "cp_tp":
        for ranks in ([0, 1], [2, 3]):
            group = dist.new_group(ranks)
            if rank in ranks:
                cp_group = group
        for ranks in ([0, 2], [1, 3]):
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_group = group
    elif topology == "cp":
        cp_group = dist.group.WORLD

    def reduce(tensor, op):
        assert threading.current_thread() is threading.main_thread()
        groups = [g for g in (cp_group, tp_group) if g is not None]
        for group in groups or [dist.group.WORLD]:
            dist.all_reduce(tensor, op=op, group=group)

    cache, wrapper = ready_lookup_cache(reduce, cp_group, tp_group, size)
    try:
        # Different queue order and missing arrivals cannot align by FIFO index.
        order = ("slow", "fast") if rank % 2 == 0 else ("fast", "slow")
        futures = {rid: controlled_probe(wrapper, rid) for rid in order}
        futures["fast"].set_result([1, 3] if rank == 0 else [2, 3])
        if rank != 0:
            futures["slow"].set_result([1])
        cache._publish_ready_external_lookups()
        assert set(wrapper.lookup_results) == {"fast"}
        assert wrapper.lookup_results["fast"][1] == 3
        assert wrapper.has_pending_lookup("slow")
        if rank == 0:
            futures["slow"].set_result([1])
        cache._publish_ready_external_lookups()
        assert wrapper.lookup_results["slow"][1] == 1

        # No query on rank 0 yet: do not consume any peer's result.
        if rank != 0:
            controlled_probe(wrapper, "late").set_result([2])
        cache._publish_ready_external_lookups()
        assert "late" not in wrapper.lookup_results
        if rank == 0:
            controlled_probe(wrapper, "late").set_result([2])
        cache._publish_ready_external_lookups()
        assert wrapper.lookup_results["late"][1] == 2

        # Equal RID/page count but different key/salt is NOT the same query.
        controlled_probe(
            wrapper, "changed", salt="old" if rank == 0 else "new"
        ).set_result([3])
        cache._publish_ready_external_lookups()
        assert "changed" not in wrapper.lookup_results
        if rank == 0:
            controlled_probe(wrapper, "changed", salt="new").set_result([1, 3])
        cache._publish_ready_external_lookups()
        assert wrapper.lookup_results["changed"][1] == 3

        # A worker exception on one rank becomes a common miss, not a hang.
        future = controlled_probe(wrapper, "error")
        if rank == 0:
            future.set_exception(RuntimeError("injected error"))
        else:
            future.set_result([3])
        cache._publish_ready_external_lookups()
        assert wrapper.lookup_results["error"][1] == 0
        assert not wrapper.lookup_queue
    finally:
        wrapper.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("topology", ["cp", "tp_fallback", "cp_tp"])
def test_dfs_ready_lookup_cross_rank_order_identity_and_errors(tmp_path, topology):
    mp.spawn(
        _ready_lookup_rank_worker,
        args=((tmp_path / topology).as_uri(), topology),
        nprocs=4 if topology == "cp_tp" else 2,
        join=True,
    )


def test_ready_lookup_identity_covers_key_and_anchor_without_container_dependence():
    signature = wrapper_class()._lookup_query_id
    key = Key(array("q", [10, 20, 30]), salt="tenant-a")
    baseline = signature(key, 0, 3)
    assert signature(Key([10, 20, 30], salt="tenant-a"), 0, 3) == baseline
    assert signature(key, 1, 2) != baseline
    assert signature(Key([10, 20, 31], salt="tenant-a"), 0, 3) != baseline
    assert signature(Key([10, 20, 30], salt="tenant-b"), 0, 3) != baseline
    key.extra_key = "adapter"
    assert signature(key, 0, 3) != baseline
