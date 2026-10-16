"""Exercise real linker methods without starting the GPU runtime."""

import ast
import logging
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"
SCOPE = {
    "logger": logging.getLogger(__name__),
    "torch": torch,
    "Future": Future,
    "arm_load_failure_injection": lambda _: lambda *_: None,
}


def load_methods(path, names):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(
                compile(
                    "from __future__ import annotations\n" + ast.unparse(node),
                    str(path),
                    "exec",
                ),
                SCOPE,
            )


load_methods(
    ROOT / "storage/mooncake_store/mooncake_direct_linker.py",
    {
        "_read_layer_layouts",
        "load_layer_wise",
        "LayerWiseLoadCounter",
        "ReadPlanLoadCounter",
        "_prepare_read_plan_layouts",
        "load_with_read_plan",
        "_prepare_expanded_load",
        "_lease_bases",
        "_rollback_session_refs_locked",
        "_abort_prepared_load_now",
        "abort_prepared_load",
    },
)
load_methods(
    ROOT / "hybrid_cache/linker_pool_assembler.py",
    {
        "get_layer_range_components",
        "get_prepared_layer_range_meta",
        "_rows",
        "prepare_locations",
        "prepare_locations_batch",
    },
)


def make_pool(packed=True):
    pool = SimpleNamespace(
        packed=packed,
        layer_mapping={0: (0, 2), 2: 1},
        buffer_meta=[[(100, 32, 4), (200, 32, 4), (300, 32, 4)]],
        _component_offsets=[[0, 4, 8]],
        prepare_locations=list,
        prepare_locations_batch=lambda batches: [
            index for indices in batches for index in indices
        ],
    )
    for name in ("get_layer_range_components", "get_prepared_layer_range_meta"):
        setattr(pool, name, MethodType(SCOPE[name], pool))
    return pool


def layout_results(layouts):
    return [
        (
            [sum(component.byte_count for component in layout.components)]
            * len(layout.keys)
            if layout.key_mapping == 1
            else [component.byte_count for component in layout.components]
            * len(layout.rows)
        )
        for layout in layouts
    ]


def make_linker(pool, store):
    linker = SimpleNamespace(
        pools={"kv": pool},
        storage=SimpleNamespace(store=store),
        _read_layout_types=(
            SimpleNamespace,
            SimpleNamespace(PER_ROW=1, PER_ROW_COMPONENT=2),
            SimpleNamespace,
        ),
        _read_layout_components={},
        read_layout_enabled=True,
        read_plan_enabled=False,
        enable_page_wise_load=False,
        num_layers=3,
        tp_rank=0,
        layer_done_counter=SCOPE["LayerWiseLoadCounter"](3),
    )
    SCOPE["time"] = time
    linker.session_lock = threading.RLock()
    linker.host_prefetch_lock = threading.RLock()
    linker.host_prefetch_enabled = False
    linker.host_prefetch_entries = {}
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    for name in (
        "_prepare_expanded_load",
        "_lease_bases",
        "_rollback_session_refs_locked",
        "_abort_prepared_load_now",
        "abort_prepared_load",
    ):
        setattr(linker, name, MethodType(SCOPE[name], linker))
    linker._read_layer_layouts = MethodType(SCOPE["_read_layer_layouts"], linker)
    linker.load_layer_wise = MethodType(SCOPE["load_layer_wise"], linker)
    linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (keys, None)
    linker.storage._tag_keys = list
    return linker


@pytest.mark.parametrize("packed", [True, False])
def test_layout_matches_legacy_draft_mapping_and_invalidates_components(packed):
    calls = []
    pool = make_pool(packed)
    linker = make_linker(
        pool,
        SimpleNamespace(
            batch_get_into_layouts=lambda layouts: calls.append(layouts)
            or layout_results(layouts)
        ),
    )
    keys = ["same"] * (2 if packed else 4)
    batches = {"kv": (keys, [7, 2])}
    linker._read_layer_layouts(batches, 0, lambda *_: None)
    layout = calls[-1][0]
    assert layout.keys == keys and layout.rows == [7, 2]
    ptrs, sizes, offsets = pool.get_prepared_layer_range_meta([7, 2], 0)
    expanded = [
        (c.destination_base + row * c.destination_stride, c.byte_count, c.source_offset)
        for row in layout.rows
        for c in layout.components
    ]
    assert expanded == [
        (p, n, o)
        for ps, ns, os in zip(ptrs, sizes, offsets)
        for p, n, o in zip(ps, ns, os)
    ]
    linker._read_layer_layouts(batches, 0, lambda *_: None)
    assert calls[-1][0].components is layout.components
    pool.buffer_meta[0][0] = (900, 32, 4)
    linker._read_layer_layouts(batches, 0, lambda *_: None)
    assert calls[-1][0].components is not layout.components
    assert calls[-1][0].components[0].destination_base == 900
    count = len(calls)
    linker._read_layer_layouts(batches, 1, lambda *_: None)
    assert len(calls) == count


def test_read_failure_preserves_published_layers_and_cleans_session():
    events = []

    def read(layouts):
        events.append("read")
        if len(events) == 2:
            raise RuntimeError("injected read failure")
        return layout_results(layouts)

    store = SimpleNamespace(
        batch_get_session_start=lambda keys: [0] * len(keys),
        batch_get_session_end=lambda keys: events.append("end") or 0,
        batch_get_into_layouts=read,
    )
    linker = make_linker(make_pool(), store)
    index = linker.layer_done_counter.update_producer()
    transfers = [[SimpleNamespace(name="kv", keys=["a"], host_indices=[2])]]
    assert linker.load_layer_wise(index, list(enumerate(transfers))) is False
    futures = linker.layer_done_counter.futures[index]
    assert futures[0].result() is None
    assert futures[1].result() is None  # Layer has no local data.
    with pytest.raises(RuntimeError, match="injected read failure"):
        futures[2].result()
    assert events == ["read", "read", "end"]


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("result_type", [list, tuple])
def test_layout_returned_byte_counts(packed, result_type):
    pool = make_pool(packed)
    pool.buffer_meta[0][2] = (300, 32, 6)

    def read(layouts):
        return result_type(
            result_type([10, 10] if packed else [4, 6, 4, 6]) for layout in layouts
        )

    linker = make_linker(pool, SimpleNamespace(batch_get_into_layouts=read))
    keys = ["a", "b"] if packed else ["a", "b", "c", "d"]
    linker._read_layer_layouts({"kv": (keys, [7, 2])}, 0, lambda *_: None)


@pytest.mark.parametrize(
    "returned",
    [None, [], [[8], [8]], [[]], [[-1]], [[4]], [[0]], [None], 8],
)
def test_layout_result_failure_does_not_publish_layer_and_cleans_session(returned):
    events = []
    store = SimpleNamespace(
        batch_get_session_start=lambda keys: [0] * len(keys),
        batch_get_session_end=lambda keys: events.append("end") or 0,
        batch_get_into_layouts=lambda _: returned,
    )
    linker = make_linker(make_pool(), store)
    index = linker.layer_done_counter.update_producer()
    transfer = SimpleNamespace(name="kv", keys=["a"], host_indices=[2])
    assert linker.load_layer_wise(index, [("req", [transfer])]) is False
    for future in linker.layer_done_counter.futures[index]:
        with pytest.raises(RuntimeError, match="Mooncake layout get"):
            future.result()
    assert events == ["end"]


def test_layout_partial_failure_preserves_earlier_completed_layer():
    events = []
    results = iter([[[8]], [[-1]]])
    store = SimpleNamespace(
        batch_get_session_start=lambda keys: [0] * len(keys),
        batch_get_session_end=lambda keys: events.append("end") or 0,
        batch_get_into_layouts=lambda _: next(results),
    )
    linker = make_linker(make_pool(), store)
    index = linker.layer_done_counter.update_producer()
    transfer = SimpleNamespace(name="kv", keys=["a"], host_indices=[2])
    assert linker.load_layer_wise(index, [("req", [transfer])]) is False
    futures = linker.layer_done_counter.futures[index]
    assert futures[0].result() is None
    assert futures[1].result() is None
    with pytest.raises(RuntimeError, match="Mooncake layout get"):
        futures[2].result()
    assert events == ["end"]


def test_partial_session_start_is_cleaned_without_reads():
    cleaned = []
    store = SimpleNamespace(
        batch_get_session_start=lambda keys: [0, -1],
        batch_get_session_end=lambda keys: cleaned.extend(keys) or 0,
        batch_get_into_layouts=lambda _: pytest.fail("unexpected I/O"),
    )
    linker = make_linker(make_pool(), store)
    index = linker.layer_done_counter.update_producer()
    assert (
        linker.load_layer_wise(
            index,
            [
                (
                    "req",
                    [SimpleNamespace(name="kv", keys=["a", "b"], host_indices=[0, 1])],
                )
            ],
        )
        is False
    )
    assert cleaned == ["a"]  # Only successfully acquired sessions need release.


@pytest.mark.parametrize("mode", ["layout", "range", "page"])
def test_request_sessions_share_keys_without_deduplicating_reads(mode):
    starts, ends, reads = [], [], []
    store = SimpleNamespace(
        batch_get_session_start=lambda keys: starts.append(list(keys))
        or [0] * len(keys),
        batch_get_session_end=lambda keys: ends.append(list(keys)) or 0,
        batch_get_into_layouts=lambda layouts: reads.extend(
            list(x.keys) for x in layouts
        )
        or layout_results(layouts),
        batch_get_into_multi_buffer_ranges=lambda keys, ptrs, sizes, offsets: (
            reads.append(list(keys)) or [sum(s) for s in sizes]
        ),
    )
    linker = make_linker(make_pool(), store)
    linker.pools["swa"] = make_pool()
    linker.read_layout_enabled = mode == "layout"
    linker.enable_page_wise_load = mode == "page"
    linker.page_wise_load_threshold = 1
    linker._load_page_wise = lambda index, batches, fail: reads.extend(
        list(keys) for keys, _ in batches.values()
    )
    transfers = [
        ("r1", [SimpleNamespace(name="kv", keys=["a"], host_indices=[0])]),
        (
            "r2",
            [
                SimpleNamespace(name="kv", keys=["a", "b"], host_indices=[1, 2]),
                SimpleNamespace(name="swa", keys=["c"], host_indices=[3]),
            ],
        ),
    ]
    index = linker.layer_done_counter.update_producer()
    assert linker.load_layer_wise(index, transfers)
    assert starts == [["a"], ["b", "c"]]
    assert ends == [["a", "b", "c"]]
    assert ["a", "a", "b"] in reads
    assert ["c"] in reads
    assert linker.prepared_load_sessions == linker.session_refcounts == {}


# Exercise admission-time session preparation with real background execution.
load_methods(
    ROOT / "storage/mooncake_store/mooncake_direct_linker.py",
    {"load", "cancel_queued_load"},
)


def admission_linker(store):
    from concurrent.futures import ThreadPoolExecutor

    linker = make_linker(make_pool(), store)
    linker.pending_loads = {}
    linker.pending_load_tokens = {}
    linker.page_size = 256
    linker.pool_group = SimpleNamespace(resolve_transfers=lambda ts, **kw: ts)
    linker.session_prepare_futures = {}
    linker.session_prepare_executor = ThreadPoolExecutor(max_workers=1)
    for name in ("load", "cancel_queued_load"):
        setattr(linker, name, MethodType(SCOPE[name], linker))
    return linker


def test_admission_returns_while_session_start_is_blocked():
    entered, release = threading.Event(), threading.Event()
    starts, ends = [], []

    def start(keys):
        starts.append(list(keys))
        entered.set()
        assert release.wait(5)
        return [0] * len(keys)

    linker = admission_linker(
        SimpleNamespace(
            batch_get_session_start=start,
            batch_get_session_end=lambda keys: ends.append(list(keys)),
        )
    )
    try:
        transfer = SimpleNamespace(keys=["shared"], name="kv")
        assert linker.load("a", [transfer])
        assert entered.wait(2)
        # Admission of b must not wait for a's RPC or its session lock.
        assert linker.load("b", [transfer])
        release.set()
        assert linker.session_prepare_futures["a"].result(5)
        assert linker.session_prepare_futures["b"].result(5)
        assert starts == [["shared"]]
        linker.abort_prepared_load("a")
        assert ends == []
        linker.abort_prepared_load("b")
        assert ends == [["shared"]]
        assert not linker.session_refcounts
    finally:
        release.set()
        linker.session_prepare_executor.shutdown(wait=True)


def test_cancel_queued_preparation_does_not_start_session():
    entered, release = threading.Event(), threading.Event()
    starts, ends = [], []

    def start(keys):
        starts.append(list(keys))
        entered.set()
        assert release.wait(5)
        return [0] * len(keys)

    linker = admission_linker(
        SimpleNamespace(
            batch_get_session_start=start,
            batch_get_session_end=lambda keys: ends.append(list(keys)),
        )
    )
    try:
        assert linker.load("a", [SimpleNamespace(keys=["a"], name="kv")])
        assert entered.wait(2)
        assert linker.load("b", [SimpleNamespace(keys=["b"], name="kv")])
        # abort also acquires the session lock, so finish a from another thread.
        timer = threading.Timer(0.05, release.set)
        timer.start()
        assert linker.cancel_queued_load("b")
        timer.join()
        assert linker.session_prepare_futures["a"].result(5)
        linker.abort_prepared_load("a")
        assert starts == [["a"]] and ends == [["a"]]
        assert "b" not in linker.pending_load_tokens
    finally:
        release.set()
        linker.session_prepare_executor.shutdown(wait=True)


def test_cancel_running_preparation_drains_before_cleanup():
    entered, release = threading.Event(), threading.Event()
    ended = []

    def start(keys):
        entered.set()
        assert release.wait(5)
        return [0] * len(keys)

    linker = admission_linker(
        SimpleNamespace(
            batch_get_session_start=start,
            batch_get_session_end=lambda keys: ended.extend(keys),
        )
    )
    try:
        assert linker.load("a", [SimpleNamespace(keys=["a"], name="kv")])
        assert entered.wait(2)
        timer = threading.Timer(0.05, release.set)
        timer.start()
        assert linker.cancel_queued_load("a")
        timer.join()
        assert ended == ["a"]
        assert not linker.prepared_load_sessions
        assert not linker.session_prepare_futures
    finally:
        release.set()
        linker.session_prepare_executor.shutdown(wait=True)


def test_failed_async_preparation_uses_load_failure_channel():
    linker = admission_linker(
        SimpleNamespace(batch_get_session_start=lambda keys: [-1] * len(keys))
    )
    try:
        transfer = SimpleNamespace(keys=["missing"], name="kv")
        assert linker.load("a", [transfer])
        counter = linker.layer_done_counter.update_producer()
        assert linker.load_layer_wise(counter, [("a", [transfer])]) is False
        assert not linker.session_prepare_futures
        assert not linker.prepared_load_sessions
        for future in linker.layer_done_counter.futures[counter]:
            assert future.done() and future.exception() is not None
    finally:
        linker.session_prepare_executor.shutdown(wait=True)


def test_distinct_admissions_start_distinct_sessions():
    starts, ends = [], []
    linker = admission_linker(
        SimpleNamespace(
            batch_get_session_start=lambda keys: starts.append(list(keys))
            or [0] * len(keys),
            batch_get_session_end=lambda keys: ends.append(list(keys)),
        )
    )
    try:
        for rid in ["a", "b"]:
            assert linker.load(rid, [SimpleNamespace(keys=[rid], name="kv")])
        for rid in ["a", "b"]:
            assert linker.session_prepare_futures[rid].result(5)
        assert starts == [["a"], ["b"]]  # Never merged into a batch start.
        for rid in ["a", "b"]:
            linker.abort_prepared_load(rid)
        assert ends == [["a"], ["b"]]
    finally:
        linker.session_prepare_executor.shutdown(wait=True)


def test_admission_reuses_claimed_dfs_session():
    ends = []
    # No start API is provided: a claimed prefetch must never start again.
    linker = admission_linker(
        SimpleNamespace(batch_get_session_end=lambda keys: ends.extend(keys))
    )
    linker.host_prefetch_enabled = True
    linker.host_prefetch_entries["a"] = {"state": "claimed"}
    linker.prepared_load_sessions["a"] = ["existing"]
    linker.session_refcounts["existing"] = 1
    try:
        assert linker.load("a", [SimpleNamespace(keys=["existing"], name="kv")])
        assert linker.session_prepare_futures["a"].result(5)
        assert linker.host_prefetch_entries["a"]["state"] == "loading"
        linker.abort_prepared_load("a")
        assert ends == ["existing"]
        assert not linker.host_prefetch_entries
    finally:
        linker.session_prepare_executor.shutdown(wait=True)


@pytest.mark.parametrize("mode", ["layout", "range", "page", "plan"])
def test_read_modes_consume_admission_sessions_without_starting_again(mode):
    starts, ends, reads = [], [], []
    linker = admission_linker(
        SimpleNamespace(
            batch_get_session_start=lambda keys: starts.append(list(keys))
            or [0] * len(keys),
            batch_get_session_end=lambda keys: ends.append(list(keys)),
            batch_get_into_layouts=lambda layouts: reads.extend(
                list(x.keys) for x in layouts
            )
            or layout_results(layouts),
            batch_get_into_multi_buffer_ranges=lambda keys, ptrs, sizes, offsets: (
                reads.append(list(keys)) or [sum(s) for s in sizes]
            ),
        )
    )
    linker.read_layout_enabled = mode == "layout"
    linker.enable_page_wise_load = mode == "page"
    linker.page_wise_load_threshold = 1
    linker._load_page_wise = lambda index, batches, fail: reads.extend(
        list(keys) for keys, _ in batches.values()
    )
    if mode == "plan":
        linker.read_plan_enabled = True
        linker.read_plan_reuse_ranges = False
        linker.layer_done_counter = SCOPE["ReadPlanLoadCounter"](3)
        for name in ("_prepare_read_plan_layouts", "load_with_read_plan"):
            setattr(linker, name, MethodType(SCOPE[name], linker))

        def create_plan(
            layouts,
            num_layers,
            *,
            reuse_ranges,
            page_wise,
            borrowed_sessions,
            buffer_owners,
        ):
            # The plan must be created after preparation, with both sessions live.
            assert set(linker.prepared_load_sessions) == {"a", "b"}
            assert linker.session_refcounts == {"a": 1, "b": 1}
            assert num_layers == 3
            assert buffer_owners is linker.pools
            assert not reuse_ranges and not page_wise and borrowed_sessions
            return SimpleNamespace(
                run=lambda: reads.extend(list(layout[0]) for layout in layouts)
            )

        linker.storage.store.create_read_plan = create_plan
    try:
        for rid in ["a", "b"]:
            assert linker.load(
                rid,
                [
                    SimpleNamespace(
                        keys=[rid], name="kv", host_indices=torch.tensor([0])
                    )
                ],
            )
        index = linker.layer_done_counter.update_producer()
        assert linker.load_layer_wise(index, list(linker.pending_loads.items()))
        assert starts == [["a"], ["b"]]
        assert ends == [["a"], ["b"]]
        assert reads == [["a", "b"]] * (1 if mode in {"page", "plan"} else 2)
        assert not linker.session_prepare_futures
        assert not linker.prepared_load_sessions
        assert not linker.session_refcounts
    finally:
        linker.session_prepare_executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "layout,enabled", [(False, False), (False, True), (True, False), (True, True)]
)
def test_session_executor_switch_is_independent_of_layout(layout, enabled):
    path = ROOT / "storage/mooncake_store/mooncake_direct_linker.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "session_prepare_executor"
            for t in n.targets
        )
    )
    created = []
    linker = SimpleNamespace(read_layout_enabled=layout)
    marker = object()

    def executor(**kwargs):
        created.append(kwargs)
        return marker

    ns = {
        "self": linker,
        "tp_rank": 0,
        "ThreadPoolExecutor": executor,
        "envs": SimpleNamespace(
            SGLANG_MOONCAKE_ASYNC_SESSION_START=SimpleNamespace(get=lambda: enabled)
        ),
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), ns)
    assert (linker.session_prepare_executor is marker) == enabled
    assert len(created) == int(enabled)


def test_disabled_async_start_prepares_at_load_time():
    starts, ends = [], []
    linker = admission_linker(
        SimpleNamespace(
            batch_get_session_start=lambda keys: starts.append(list(keys))
            or [0] * len(keys),
            batch_get_session_end=lambda keys: ends.append(list(keys)),
            batch_get_into_layouts=layout_results,
        )
    )
    linker.session_prepare_executor.shutdown(wait=True)
    linker.session_prepare_executor = None
    assert linker.load("a", [SimpleNamespace(keys=["a"], name="kv", host_indices=[0])])
    assert not starts and not linker.session_prepare_futures
    index = linker.layer_done_counter.update_producer()
    assert linker.load_layer_wise(index, list(linker.pending_loads.items()))
    assert starts == ends == [["a"]]


def make_index_pool(packed=True, row_span=1):
    pool = make_pool(packed)
    pool.name = "kv"
    pool.page_size = 4
    pool._page_offsets = torch.arange(4)
    pool._row_span = row_span
    pool._row_count = 32
    pool.kv_buffer = [torch.empty(32, 1)]
    for name in ("_rows", "prepare_locations", "prepare_locations_batch"):
        setattr(pool, name, MethodType(SCOPE[name], pool))
    return pool


@pytest.mark.parametrize("row_span", [1, 4])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_batched_locations_preserve_order_duplicates_and_conversion(
    row_span, mixed_dtype
):
    pool = make_index_pool(row_span=row_span)
    inputs = [
        torch.tensor([[8, 9], [10, 11]]),
        torch.empty(0, dtype=torch.int64),
        torch.tensor(
            [0, 1, 2, 3, 8, 9, 10, 11],
            dtype=torch.int32 if mixed_dtype else torch.int64,
        ),
    ]
    expected = [row for item in inputs for row in pool.prepare_locations(item)]
    assert pool.prepare_locations_batch(inputs) == expected
    assert pool.prepare_locations_batch([]) == []
    assert pool.prepare_locations_batch(inputs[:1]) == pool.prepare_locations(inputs[0])


@pytest.mark.parametrize(
    "indices",
    [
        [[0, 1], [2, 3]],  # Concatenation must not repair invalid transfer boundaries.
        [[1, 2, 3, 4]],
        [[0, 1, 3, 2]],
        [[-4, -3, -2, -1]],
        [[128, 129, 130, 131]],
    ],
)
def test_batched_locations_reject_invalid_indices(indices):
    pool = make_index_pool()
    with pytest.raises(ValueError):
        pool.prepare_locations_batch([torch.tensor(item) for item in indices])


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_batched_locations_linker_equivalence_and_cleanup(packed, invalid):
    results = []
    for batched in (False, True):
        calls, ended = [], []
        store = SimpleNamespace(
            batch_get_session_start=lambda keys: [0] * len(keys),
            batch_get_session_end=lambda keys: ended.extend(keys) or 0,
            batch_get_into_layouts=lambda layouts: calls.append(layouts)
            or layout_results(layouts),
        )
        linker = make_linker(make_index_pool(packed), store)
        linker.pools["second"] = make_index_pool(packed)
        if not batched:
            # Compare against per-transfer preparation without a runtime switch.
            for pool in linker.pools.values():
                pool.prepare_locations_batch = lambda indices, pool=pool: [
                    row for item in indices for row in pool.prepare_locations(item)
                ]
        if not packed:
            # Unpacked pools use one key per row/component on every layer.
            for pool in linker.pools.values():
                pool.layer_mapping[2] = (0, 2)
            linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
                [f"{key}_{component}" for key in keys for component in range(2)],
                None,
            )
        index = linker.layer_done_counter.update_producer()
        requests = [
            (
                str(i),
                [
                    SimpleNamespace(
                        name=name,
                        keys=[str(i) + name],
                        host_indices=torch.tensor(
                            [0, 1] if invalid else [i * 4 + j for j in range(4)]
                        ),
                    )
                    for name in ("kv", "second")
                ],
            )
            for i in range(2)
        ]
        ok = linker.load_layer_wise(index, requests)
        assert ok is not invalid
        expected_keys = ["0kv", "0second", "1kv", "1second"]
        if not packed:
            expected_keys = [
                f"{key}_{component}" for key in expected_keys for component in range(2)
            ]
        assert sorted(ended) == expected_keys
        assert linker.prepared_load_sessions == {}
        if invalid:
            assert calls == []
            with pytest.raises(RuntimeError, match="ValueError: Pool"):
                linker.layer_done_counter.futures[index][0].result()
        results.append(
            [[(layout.keys, layout.rows) for layout in call] for call in calls]
        )
    assert results[0] == results[1]


def test_layout_batch_locations_do_not_change_range_loads():
    converted = []
    pool = make_index_pool()
    prepare = pool.prepare_locations
    pool.prepare_locations = lambda indices: converted.append(
        indices.tolist()
    ) or prepare(indices)
    pool.prepare_locations_batch = lambda _: pytest.fail("unexpected batch preparation")
    store = SimpleNamespace(
        batch_get_session_start=lambda keys: [0] * len(keys),
        batch_get_session_end=lambda keys: 0,
        batch_get_into_multi_buffer_ranges=lambda keys, ptrs, sizes, offsets: [
            sum(item) for item in sizes
        ],
    )
    linker = make_linker(pool, store)
    linker.read_layout_enabled = False
    index = linker.layer_done_counter.update_producer()
    requests = [
        (
            str(i),
            [
                SimpleNamespace(
                    name="kv",
                    keys=[str(i)],
                    host_indices=torch.arange(i * 4, i * 4 + 4),
                )
            ],
        )
        for i in range(2)
    ]
    assert linker.load_layer_wise(index, requests)
    assert converted == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert all(
        future.result() is None for future in linker.layer_done_counter.futures[index]
    )
