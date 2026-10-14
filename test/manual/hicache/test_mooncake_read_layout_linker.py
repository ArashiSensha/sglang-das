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
