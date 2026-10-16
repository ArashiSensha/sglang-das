"""Scheduler-level regressions for async lookup collectives and query identity."""

import ast
from array import array
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum, auto
from types import SimpleNamespace as NS
from typing import Union

import pytest
import test_mooncake_async_lookup as support
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class StopAdmission(Exception):
    pass


def scheduler_fixture(queue, *, dfs=True, poll=None, full=False, capacity=2):
    events = []

    class Adder:
        def __init__(self, *args, **kwargs):
            self.can_run_list = []

        def add_one_req(self, req, **kwargs):
            raise StopAdmission

    cls = support.extracted_methods(
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
    cache = NS(linker=NS(async_lookup_enabled=True))
    cache.submit_external_lookups = (
        lambda reqs: events.append(("submit", len(reqs))) or True
    )

    def check():
        events.append(("poll",))
        if poll is not None:
            poll()

    cache.check_hicache_events = check
    cache.is_external_lookup_pending = lambda rid: False
    cache.get_waiting_queue_prefetch_admission_states = lambda rids, **kw: {}
    scheduler = cls()
    scheduler.tree_cache = cache
    scheduler.grammar_manager = NS(has_waiting_grammars=lambda: False)
    scheduler.waiting_queue = queue
    scheduler.enable_waiting_queue_dfs_prefetch = dfs
    scheduler._complete_waiting_queue_prefetch_lookups = lambda: []
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_unified_cache_external_linker = True
    scheduler.enable_priority_preemption = scheduler.is_hybrid_swa = False
    scheduler.chunked_req = scheduler.min_free_slots_delayer = None
    scheduler.get_num_allocatable_reqs = lambda bs: capacity
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
    scheduler.req_to_token_pool = NS(available_size=lambda: capacity)
    scheduler.disaggregation_mode = "prefill"
    scheduler.truncation_align_size = None
    batch = NS(batch_is_full=full, reqs=[])
    return scheduler, batch, events


def run_scheduler(scheduler, batch):
    try:
        scheduler._get_new_batch_prefill_raw(None, batch)
    except StopAdmission:
        pass


@pytest.mark.parametrize("dfs", [False, True])
@pytest.mark.parametrize("early_return", ["empty", "full", "capacity", "delayed"])
def test_unequal_waiting_queues_have_one_event_sync(dfs, early_return):
    req = NS(rid="queued", init_next_round_input=lambda cache: None)
    left_queue = [] if early_return == "empty" else [req]
    left = scheduler_fixture(
        left_queue,
        dfs=dfs,
        full=early_return == "full",
        capacity=0 if early_return == "capacity" else 2,
    )
    if early_return == "delayed":
        left[0].min_free_slots_delayer = NS(should_delay=lambda **kwargs: True)
    right = scheduler_fixture([req], dfs=dfs)
    for scheduler, batch, _ in (left, right):
        run_scheduler(scheduler, batch)
    assert [sum(e[0] == "poll" for e in item[2]) for item in (left, right)] == [1, 1]


def _unequal_queue_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        req = NS(rid="bootstrap-to-waiting", init_next_round_input=lambda cache: None)
        calls = []

        def poll():
            calls.append(1)
            dist.all_reduce(torch.tensor([1], dtype=torch.int))

        scheduler, batch, _ = scheduler_fixture([] if rank == 0 else [req], poll=poll)
        # The request is still in rank zero's bootstrap queue.
        scheduler.disagg_prefill_bootstrap_queue = NS(queue=[req] if rank == 0 else [])
        run_scheduler(scheduler, batch)
        counts = [None, None]
        dist.all_gather_object(counts, len(calls))
        assert counts == [1, 1]
    finally:
        dist.destroy_process_group()


def test_unequal_bootstrap_waiting_queues_do_not_strand_collectives(tmp_path):
    mp.spawn(
        _unequal_queue_rank,
        args=((tmp_path / "unequal").as_uri(),),
        nprocs=2,
        join=True,
    )


def policy_and_request(wrapper, token_count):
    key_cls = support.extracted_methods(
        "python/sglang/srt/mem_cache/radix_cache.py",
        "RadixKey",
        {"__init__", "_raw_len", "raw_token_ids", "__len__", "__getitem__"},
        {"array": array},
    )
    req_cls = support.extracted_methods(
        "python/sglang/srt/managers/schedule_batch.py",
        "Req",
        {
            "prepare_external_lookup",
            "_prefix_match_params",
            "_refresh_fill_ids",
            "_compute_max_prefix_len",
        },
        {"array": array, "RadixKey": key_cls, "MatchPrefixParams": NS},
    )
    req = req_cls()
    req.rid = "review-miss"
    req.origin_input_ids, req.output_ids = array("q", range(token_count)), array("q")
    req.full_untruncated_fill_ids = array("q")
    req.external_cache_hit_length = req.session = req.pd_hidden_max_prefix_len = None
    req.positional_embed_overrides = None
    req.is_dllm = lambda: False
    req.return_logprob, req.logprob_start_len = False, -1
    req.extra_key, req.cache_salt = "adapter", "tenant"
    fields = (
        "device_indices last_device_node best_match_node last_host_node host_hit_length "
        "swa_host_hit_length mamba_host_hit_length mamba_branching_seqlen cache_protected_len"
    )
    from collections import namedtuple

    match_type = namedtuple("PolicyMatch", fields)
    local = match_type(
        torch.empty(0, dtype=torch.int64), 0, 0, None, 0, 0, 0, None, None
    )
    cache = wrapper.cache
    cache.linker = wrapper
    cache.swa_reprefill_tail_tokens = lambda: 0
    cache.match_prefix = lambda params: wrapper.match(params.key, params.req, local)
    cache.supports_fast_match_prefix = lambda: False

    class Node:
        children = {}

    cache.root_node = Node()
    cache.resolve_node_handle = lambda node: cache.root_node
    source = support.SOURCE.parents[5] / "python/sglang/srt/managers/schedule_policy.py"
    tree = ast.parse(source.read_text())
    names = {
        "match_prefix_for_req",
        "CacheAwarePolicy",
        "CacheAgnosticPolicy",
        "SchedulePolicy",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    ns = {
        "array": array,
        "torch": torch,
        "Enum": Enum,
        "auto": auto,
        "Union": Union,
        "contextmanager": contextmanager,
        "defaultdict": defaultdict,
        "RadixKey": key_cls,
        "MatchPrefixParams": NS,
        "envs": NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
        "get_disagg": lambda: NS(disaggregation_mode="prefill"),
        "IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD": -1,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    policy = ns["SchedulePolicy"].__new__(ns["SchedulePolicy"])
    policy.tree_cache = cache
    policy.waiting_queue_radix_tree = NS(reset=lambda: None)
    policy.enable_priority_scheduling = False
    return policy, req, ns


@pytest.mark.parametrize("name", ["lpm", "dfs-weight"])
@pytest.mark.parametrize("token_count", [11, 12])
@pytest.mark.parametrize("hit", [False, True])
def test_cache_aware_policy_does_not_replace_admission_lookup(name, token_count, hit):
    cache, wrapper = support.ready_lookup_cache()
    wrapper.cache.page_size = 4
    wrapper.num_completed_loads = lambda: 0
    wrapper.num_completed_offloads = lambda: 0
    wrapper._tail_hashes = lambda key, result, device_hit_len: [
        str(i) for i in range(device_hit_len // 4, len(key) // 4)
    ]
    policy, req, ns = policy_and_request(wrapper, token_count)
    policy.policy = ns["CacheAwarePolicy"](name)
    cache._collect_failed_linker_loads = lambda count: None
    cache.match_prefix = wrapper.cache.match_prefix
    cache.linker = wrapper
    try:
        params = req.prepare_external_lookup(wrapper.cache)
        wrapper.cache.match_prefix(params)
        admitted = False
        for _ in range(4):
            pending = wrapper.pending_lookups.get(req.rid)
            if pending is not None and not pending.future.done():
                pending.future.set_result([1, 2] if hit else [])
            cache.check_hicache_events()
            policy.calc_priority([req])
            wrapper.cache.match_prefix(req.prepare_external_lookup(wrapper.cache))
            if not wrapper.has_pending_lookup(req.rid):
                admitted = True
                break
        assert admitted, "priority match keeps replacing the admission probe"
        assert len(wrapper.lookup_executor.futures) == 1
        assert req.host_hit_length == (8 if hit else 0)
    finally:
        wrapper.close()


@pytest.mark.parametrize(
    "logprob_cap,hidden_cap,swa_tail,overrides,expected_len",
    [
        (None, None, 0, False, 11),
        (4, None, 0, False, 4),
        (None, 5, 0, False, 5),
        (None, None, 3, False, 9),
        (4, 5, 3, False, 4),
        (None, None, 0, True, 0),
    ],
)
def test_priority_and_admission_use_identical_query_parameters(
    logprob_cap, hidden_cap, swa_tail, overrides, expected_len
):
    _, wrapper = support.ready_lookup_cache()
    try:
        _, req, ns = policy_and_request(wrapper, 12)
        req.return_logprob = logprob_cap is not None
        req.logprob_start_len = logprob_cap if logprob_cap is not None else -1
        req.pd_hidden_max_prefix_len = hidden_cap
        req.positional_embed_overrides = object() if overrides else None
        wrapper.cache.swa_reprefill_tail_tokens = lambda: swa_tail
        canonical = req.prepare_external_lookup(wrapper.cache)
        captured = []
        match = wrapper.cache.match_prefix
        wrapper.cache.match_prefix = lambda params: captured.append(params) or match(
            params
        )
        ns["match_prefix_for_req"](wrapper.cache, req, cow_mamba=True, include_req=True)
        actual = captured[0]
        assert list(actual.key.raw_token_ids()) == list(canonical.key.raw_token_ids())
        assert len(actual.key) == expected_len
        assert actual.key.extra_key == canonical.key.extra_key == "adapter"
        assert actual.key.cache_salt == canonical.key.cache_salt == "tenant"
        assert actual.return_full_match == canonical.return_full_match == bool(swa_tail)
        assert actual.cow_mamba
        assert actual.req is req
    finally:
        wrapper.close()


@pytest.mark.parametrize("async_enabled,include_req", [(False, True), (True, False)])
def test_other_priority_paths_keep_existing_full_input_match(
    async_enabled, include_req
):
    _, wrapper = support.ready_lookup_cache()
    try:
        _, req, ns = policy_and_request(wrapper, 12)
        wrapper.async_lookup_enabled = async_enabled
        req.prepare_external_lookup = lambda cache: pytest.fail(
            "unexpected async staging"
        )
        captured = []
        local = NS(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=0,
            last_host_node=None,
            best_match_node=0,
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=None,
            cache_protected_len=None,
        )
        wrapper.cache.match_prefix = lambda params: captured.append(params) or local
        ns["match_prefix_for_req"](wrapper.cache, req, include_req=include_req)
        actual = captured[0]
        assert list(actual.key.raw_token_ids()) == list(req.origin_input_ids)
        assert actual.req is (req if include_req else None)
    finally:
        wrapper.close()


@pytest.mark.parametrize("operation", ["capacity", "slice"])
def test_disabled_lookup_skips_speculative_candidate_work(operation):
    class UnslicedQueue(list):
        def __getitem__(self, key):
            if isinstance(key, slice):
                pytest.fail("disabled lookup sliced the waiting queue")
            return super().__getitem__(key)

    scheduler, batch, _ = scheduler_fixture(UnslicedQueue(), dfs=False)
    scheduler.tree_cache.linker.async_lookup_enabled = False
    if operation == "capacity":
        scheduler.get_num_allocatable_reqs = lambda _: pytest.fail(
            "disabled lookup computed speculative capacity"
        )
    run_scheduler(scheduler, batch)


def test_disabled_lookup_skips_dfs_completion_scan_but_keeps_state_sync():
    scheduler, batch, events = scheduler_fixture([], dfs=True)
    scheduler.tree_cache.linker.async_lookup_enabled = False
    scheduler._complete_waiting_queue_prefetch_lookups = lambda: pytest.fail(
        "disabled lookup scanned deferred DFS probes"
    )
    scheduler.tree_cache.get_waiting_queue_prefetch_admission_states = (
        lambda rids, **kwargs: events.append(("dfs_states",)) or {}
    )
    run_scheduler(scheduler, batch)
    assert events == [("poll",), ("dfs_states",)]


def test_disabled_lookup_completion_does_not_iterate_ingress_queues():
    class UnscannedQueue(list):
        def __iter__(self):
            pytest.fail("disabled lookup iterated an ingress queue")

    wrapper, _ = support.make_wrapper()
    wrapper.async_lookup_enabled = False
    scheduler = support.prefetch_scheduler(wrapper)
    scheduler.waiting_queue = UnscannedQueue()
    scheduler.disagg_prefill_bootstrap_queue.queue = UnscannedQueue()
    try:
        assert scheduler._complete_waiting_queue_prefetch_lookups() == []
    finally:
        wrapper.close()


def test_disabled_lookup_events_keep_original_two_counts_and_offload_sync():
    vectors, events = [], []
    cache, wrapper = support.ready_lookup_cache(
        lambda counts, op: vectors.append(counts.tolist())
    )
    wrapper.async_lookup_enabled = False
    wrapper.num_completed_loads = lambda: 2
    wrapper.num_completed_offloads = lambda: 1
    wrapper.wait_pending_lookups = lambda: events.append("wait_lookup") or 0
    wrapper.drain_lookups = lambda count: events.append("drain_lookup")
    wrapper.take_completed_offloads = lambda count: [True]
    wrapper.commit_completed_offloads = lambda successes: events.append(
        ("offload", successes)
    )
    cache._collect_failed_linker_loads = lambda count: events.append(("load", count))
    try:
        cache.check_hicache_events()
        assert vectors == [[2, 1], [1]]
        assert events == [("load", 2), ("offload", [True])]
    finally:
        wrapper.close()


def _disabled_lookup_events_rank(rank, rendezvous, dfs):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    wrapper = None
    try:

        def reduce(counts, op):
            assert counts.numel() == 2
            dist.all_reduce(counts, op=op)

        cache, wrapper = support.ready_lookup_cache(reduce)
        wrapper.async_lookup_enabled = False
        wrapper.cache_linker.waiting_queue_prefetch_enabled = lambda: dfs
        wrapper.num_completed_loads = lambda: 2 - rank
        wrapper.num_completed_offloads = lambda: rank
        wrapper.wait_pending_lookups = lambda: pytest.fail("disabled lookup waited")
        wrapper.drain_lookups = lambda count: pytest.fail("disabled lookup drained")
        drained = []
        cache._collect_failed_linker_loads = lambda count: drained.append(count)
        wrapper.take_completed_offloads = lambda count: []
        cache.check_hicache_events()
        counts = [None, None]
        dist.all_gather_object(counts, drained)
        assert counts == [[1], [1]]
    finally:
        if wrapper is not None:
            wrapper.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("dfs", [False, True])
def test_disabled_lookup_two_rank_events_keep_matching_collectives(tmp_path, dfs):
    mp.spawn(
        _disabled_lookup_events_rank,
        args=((tmp_path / "disabled").as_uri(), dfs),
        nprocs=2,
        join=True,
    )
