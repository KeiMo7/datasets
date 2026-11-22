# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
"""Tests for schema-aware shard merging (`datasets.merge`).

These tests cover the eight shard shapes and the non-negotiable guarantees of
the merge design: auditable per-column promotions, class-label text preserved
row by row, identical semantics across the map-style and streaming paths,
disk round-trips, lossless-only enforcement and readable incompatibility errors.
"""

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasets import (
    ClassLabel,
    Dataset,
    Features,
    List,
    MergeError,
    MergeReport,
    MergeStrategy,
    Value,
    concatenate_datasets,
    load_from_disk,
)
from datasets.merge import (
    unify_example,
    unify_features,
    unify_table,
)


# ---------------------------------------------------------------------------
# Schema-only unification
# ---------------------------------------------------------------------------


def test_unify_integer_widths_is_lossless():
    # Shape 1: same column, different integer widths.
    unified, report = unify_features([Features({"a": Value("int32")}), Features({"a": Value("int64")})])
    assert unified == Features({"a": Value("int64")})
    assert not report.lossy
    column = report.column_report("a")
    assert column.kind == "promotion"
    assert (0, "Value('int32')", "Value('int64')", False) in column.conversions


def test_unify_float_precision_with_missing_values():
    # Shape 2: float precision difference (missing values live in the data).
    unified, report = unify_features([Features({"f": Value("float32")}), Features({"f": Value("float64")})])
    assert unified == Features({"f": Value("float64")})
    assert not report.lossy


def test_unify_class_label_vs_string():
    # Shape 3: a class dictionary on one side, plain strings on the other.
    unified, report = unify_features(
        [
            Features({"label": ClassLabel(names=["a", "b"])}),
            Features({"label": Value("string")}),
        ]
    )
    assert isinstance(unified["label"], ClassLabel)
    assert unified["label"].names == ["a", "b"]


def test_unify_class_labels_different_names_and_order():
    # Shape 4: distinct class-name sets and distinct ordering.
    unified, report = unify_features(
        [
            Features({"label": ClassLabel(names=["a", "b", "c"])}),
            Features({"label": ClassLabel(names=["c", "d", "a"])}),
        ]
    )
    merged = unified["label"]
    assert isinstance(merged, ClassLabel)
    # Union, in first-seen order.
    assert merged.names == ["a", "b", "c", "d"]
    assert report.column_report("label").kind == "class_label_merge"
    assert not report.lossy


def test_unify_missing_and_extra_columns_are_null_filled():
    # Shape 5: one shard carries an extra column.
    unified, report = unify_features(
        [
            Features({"a": Value("int64")}),
            Features({"a": Value("int64"), "b": Value("string")}),
        ]
    )
    assert set(unified) == {"a", "b"}
    assert report.column_report("b").kind == "missing_column"
    assert report.column_report("b").source_features[0] is None


def test_unify_null_only_column_adopts_real_type():
    # The all-empty timestamp bucket must adopt the real type, not stay null.
    unified, report = unify_features(
        [
            Features({"ts": Value("null")}),
            Features({"ts": Value("timestamp[ns]")}),
        ]
    )
    assert unified == Features({"ts": Value("timestamp[ns]")})
    assert not report.lossy


def test_unify_incompatible_type_families_raise_readable_error():
    # Failure case 1: string vs numeric, naming the column and both types.
    with pytest.raises(MergeError) as exc_info:
        unify_features([Features({"a": Value("string")}), Features({"a": Value("int64")})])
    error = exc_info.value
    assert error.column_name == "a"
    rendered = str(error)
    assert "string" in rendered and "int64" in rendered


def test_unify_structurally_different_columns_raise():
    with pytest.raises(MergeError) as exc_info:
        unify_features(
            [
                Features({"a": Value("int64")}),
                Features({"a": {"x": Value("int64")}}),
            ]
        )
    assert exc_info.value.column_name == "a"


def test_lossless_strategy_rejects_lossy_promotion():
    # Guarantee 4: declaring "lossless only" refuses lossy columns.
    with pytest.raises(MergeError):
        unify_features(
            [Features({"a": Value("float64")}), Features({"a": Value("int64")})],
            MergeStrategy(allow_lossy=False),
        )
    unified, report = unify_features(
        [Features({"a": Value("float64")}), Features({"a": Value("int64")})],
        MergeStrategy(allow_lossy=True),
    )
    assert report.lossy
    assert unified == Features({"a": Value("float64")})


def test_disabling_missing_column_fill_raises():
    with pytest.raises(MergeError):
        unify_features(
            [
                Features({"a": Value("int64")}),
                Features({"a": Value("int64"), "b": Value("string")}),
            ],
            MergeStrategy(fill_missing_columns=False),
        )


def test_report_covers_every_column():
    # Acceptance: the promotion report covers every column.
    _, report = unify_features(
        [
            Features({"a": Value("int32"), "b": Value("string"), "c": Value("float32")}),
            Features({"a": Value("int64"), "b": Value("large_string"), "c": Value("float64")}),
        ]
    )
    assert {column.column_name for column in report.column_reports} == {"a", "b", "c"}
    assert all(not column.lossy for column in report.column_reports)


def test_identical_features_have_no_promotions():
    # Existing behavior 1: identical schemas stay identical, no promotions.
    features = Features({"a": Value("int64")})
    unified, report = unify_features([features, features])
    assert unified == features
    assert not report.has_promotions
    assert report.column_report("a").is_identical


# ---------------------------------------------------------------------------
# Data-level unification
# ---------------------------------------------------------------------------


def _heterogeneous_shards():
    f1 = Features(
        {
            "id": Value("int32"),
            "label": ClassLabel(names=["a", "b", "c"]),
            "f": Value("float32"),
        }
    )
    f2 = Features(
        {
            "id": Value("int64"),
            "label": ClassLabel(names=["c", "b", "a"]),
            "f": Value("float64"),
        }
    )
    f3 = Features(
        {
            "id": Value("int64"),
            "label": ClassLabel(names=["a", "c"]),
            "extra": Value("string"),
            "f": Value("float64"),
        }
    )
    d1 = Dataset.from_dict({"id": [1, 2], "label": [0, 1], "f": [1.0, 2.0]}, features=f1)
    d2 = Dataset.from_dict({"id": [3, 4], "label": [0, 1], "f": [3.5, None]}, features=f2)
    d3 = Dataset.from_dict({"id": [5, 6], "label": [0, 1], "extra": ["x", "y"], "f": [5.0, 6.0]}, features=f3)
    return [d1, d2, d3]


EXPECTED_LABEL_TEXTS = ["a", "b", "c", "b", "a", "c"]
EXPECTED_IDS = [1, 2, 3, 4, 5, 6]
EXPECTED_EXTRA = [None, None, None, None, "x", "y"]
EXPECTED_F = [1.0, 2.0, 3.5, None, 5.0, 6.0]


def test_unify_table_reindexes_class_labels_row_by_row():
    shards = _heterogeneous_shards()
    unified, report = unify_features([d.features for d in shards])
    tables = [unify_table(d.data.table, index, unified, report) for index, d in enumerate(shards)]
    merged = pa.concat_tables(tables)
    label_feature = unified["label"]
    texts = [label_feature.int2str(v) for v in merged.column("label").to_pylist()]
    assert texts == EXPECTED_LABEL_TEXTS
    # Missing column is null-filled, never zero-filled and no row is dropped.
    assert merged.column("extra").to_pylist() == EXPECTED_EXTRA
    assert merged.num_rows == 6


def test_unify_example_matches_table_semantics():
    shards = _heterogeneous_shards()
    unified, report = unify_features([d.features for d in shards])
    rows = []
    for index, d in enumerate(shards):
        for example in d.to_list():
            rows.append(unify_example(example, index, unified, report))
    label_feature = unified["label"]
    texts = [label_feature.int2str(row["label"]) for row in rows]
    assert texts == EXPECTED_LABEL_TEXTS
    assert [row["id"] for row in rows] == EXPECTED_IDS
    assert [row["extra"] for row in rows] == EXPECTED_EXTRA


# ---------------------------------------------------------------------------
# Map-style concatenation path
# ---------------------------------------------------------------------------


def test_map_style_concatenation_preserves_rows_and_labels():
    merged = concatenate_datasets(_heterogeneous_shards())
    # Acceptance: row count is the sum (no rows lost).
    assert merged.num_rows == 6
    label_feature = merged.features["label"]
    texts = [label_feature.int2str(v) for v in merged["label"]]
    # Acceptance: label texts match row by row.
    assert texts == EXPECTED_LABEL_TEXTS
    assert merged["id"] == EXPECTED_IDS
    assert merged["extra"] == EXPECTED_EXTRA
    assert merged["f"] == EXPECTED_F
    assert isinstance(merged.merge_report, MergeReport)


def test_map_style_null_timestamp_bucket_is_recovered():
    d1 = Dataset.from_dict({"ts": [None, None]}, features=Features({"ts": Value("null")}))
    d2 = Dataset.from_dict(
        {"ts": [datetime.datetime(2024, 1, 1)]},
        features=Features({"ts": Value("timestamp[ns]")}),
    )
    merged = concatenate_datasets([d1, d2])
    assert merged.features["ts"] == Value("timestamp[ns]")
    assert merged["ts"][0] is None and merged["ts"][-1] is not None


def test_map_style_incompatible_raises_with_column_and_types():
    d1 = Dataset.from_dict({"a": ["x"]})
    d2 = Dataset.from_dict({"a": [1]}, features=Features({"a": Value("int64")}))
    with pytest.raises(MergeError) as exc_info:
        concatenate_datasets([d1, d2])
    message = str(exc_info.value)
    assert exc_info.value.column_name == "a"
    assert "string" in message and "int64" in message


def test_map_style_round_trip_disk(tmp_path):
    # Acceptance: save to disk and read back keeps types and labels.
    merged = concatenate_datasets(_heterogeneous_shards())
    merged.save_to_disk(tmp_path / "merged")
    reloaded = load_from_disk(tmp_path / "merged")
    assert reloaded.features == merged.features
    assert reloaded.num_rows == merged.num_rows
    label_feature = reloaded.features["label"]
    assert [label_feature.int2str(v) for v in reloaded["label"]] == EXPECTED_LABEL_TEXTS
    assert reloaded["id"] == EXPECTED_IDS


# ---------------------------------------------------------------------------
# Streaming concatenation path and cross-path equivalence
# ---------------------------------------------------------------------------


def test_streaming_concatenation_matches_map_style():
    shards = _heterogeneous_shards()
    map_merged = concatenate_datasets(shards)

    streamed = concatenate_datasets([d.to_iterable_dataset() for d in shards])
    rows = list(streamed)

    # Acceptance: streaming and in-memory paths give identical results.
    assert len(rows) == map_merged.num_rows
    stream_label = streamed.features["label"]
    map_label = map_merged.features["label"]
    assert [stream_label.int2str(r["label"]) for r in rows] == [map_label.int2str(v) for v in map_merged["label"]]
    assert [r["id"] for r in rows] == map_merged["id"]
    assert [r["extra"] for r in rows] == map_merged["extra"]
    assert [r["f"] for r in rows] == map_merged["f"]
    assert streamed.features == map_merged.features
    assert isinstance(streamed.merge_report, MergeReport)


def test_streaming_arrow_batches_are_cast_lazily(tmp_path):
    # Shape 7/8: stream shards file by file and persist without loading all data.
    shards = _heterogeneous_shards()
    parquet_paths = []
    for index, d in enumerate(shards):
        path = tmp_path / f"shard_{index}.parquet"
        d.to_parquet(path)
        parquet_paths.append(path)

    from datasets import load_dataset

    streamed_shards = [
        load_dataset("parquet", data_files=str(path), split="train", streaming=True) for path in parquet_paths
    ]
    streamed = concatenate_datasets(streamed_shards)
    output = tmp_path / "out.parquet"
    streamed.to_parquet(output)

    table = pq.read_table(output)
    assert table.num_rows == 6
    reloaded_features = Features.from_arrow_schema(table.schema)
    label_feature = reloaded_features["label"]
    texts = [label_feature.int2str(v) for v in table.column("label").to_pylist()]
    assert texts == EXPECTED_LABEL_TEXTS


def test_streaming_incompatible_raises_upfront():
    # The schema check happens before any row is consumed.
    d1 = Dataset.from_dict({"a": ["x"]}).to_iterable_dataset()
    d2 = Dataset.from_dict({"a": [1]}, features=Features({"a": Value("int64")})).to_iterable_dataset()
    with pytest.raises(MergeError):
        concatenate_datasets([d1, d2])


# ---------------------------------------------------------------------------
# Fingerprint / cache identity follows the strategy
# ---------------------------------------------------------------------------


def test_fingerprint_changes_with_merge_strategy():
    d1 = Dataset.from_dict({"a": [1]}, features=Features({"a": Value("int32")}))
    d2 = Dataset.from_dict({"a": [2]}, features=Features({"a": Value("int64")}))
    default_fp = concatenate_datasets([d1, d2])._fingerprint
    explicit_fp = concatenate_datasets([d1, d2], merge_strategy=MergeStrategy(allow_lossy=False))._fingerprint
    # Default strategy equals the explicit lossless strategy.
    assert default_fp == explicit_fp

    d3 = Dataset.from_dict({"a": [1]})
    d4 = Dataset.from_dict({"a": [2], "b": ["x"]})
    fill_fp = concatenate_datasets([d3, d4], merge_strategy=MergeStrategy(fill_missing_columns=True))._fingerprint
    assert fill_fp != default_fp


def test_nested_list_and_struct_unification():
    f1 = Features({"tags": List(Value("int32")), "obj": {"x": Value("int32")}})
    f2 = Features({"tags": List(Value("int64")), "obj": {"x": Value("int64")}})
    merged = concatenate_datasets(
        [
            Dataset.from_dict({"tags": [[1], [2, 3]], "obj": [{"x": 1}, {"x": 2}]}, features=f1),
            Dataset.from_dict({"tags": [[4]], "obj": [{"x": 3}]}, features=f2),
        ]
    )
    assert merged.features["tags"] == List(Value("int64"))
    assert merged["tags"] == [[1], [2, 3], [4]]
    assert merged["obj"] == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_unsigned_and_signed_integer_merge():
    # uint32 + int32 needs int64 to stay lossless; uint64 + int64 has no lossless home.
    unified, report = unify_features([Features({"a": Value("uint32")}), Features({"a": Value("int32")})])
    assert unified == Features({"a": Value("int64")})
    assert not report.lossy

    with pytest.raises(MergeError):
        unify_features([Features({"a": Value("uint64")}), Features({"a": Value("int64")})])


def test_memory_mapped_shards_keep_lazy_replay(tmp_path):
    # Requirement 7/8: large on-disk shards are not pulled entirely into RAM;
    # the unification is recorded as a replay and survives pickling.
    import pickle

    from datasets.table import MemoryMappedTable

    f1 = Features({"id": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    f2 = Features({"id": Value("int64"), "label": ClassLabel(names=["b", "a"])})
    d1 = Dataset.from_dict({"id": list(range(100)), "label": [i % 2 for i in range(100)]}, features=f1)
    d2 = Dataset.from_dict({"id": list(range(100, 200)), "label": [i % 2 for i in range(100)]}, features=f2)
    d1.save_to_disk(tmp_path / "d1")
    d2.save_to_disk(tmp_path / "d2")
    m1, m2 = load_from_disk(tmp_path / "d1"), load_from_disk(tmp_path / "d2")

    merged = concatenate_datasets([m1, m2])
    block_types = {type(block).__name__ for row in merged._data.blocks for block in row}
    assert "MemoryMappedTable" in block_types
    replay_ops = [
        replay[0]
        for row in merged._data.blocks
        for block in row
        if isinstance(block, MemoryMappedTable)
        for replay in block.replays
    ]
    assert "merge_unify" in replay_ops

    restored = pickle.loads(pickle.dumps(merged))
    assert restored.num_rows == 200
    label_feature = restored.features["label"]
    assert label_feature.int2str(restored["label"][0]) == "a"
    assert label_feature.int2str(restored["label"][100]) == "b"


def test_merge_report_answers_can_merge_and_lossiness():
    _, report = unify_features([Features({"a": Value("int32")}), Features({"a": Value("int64")})])
    # Guarantee 5: the report directly answers the two merge questions.
    assert report.can_merge() is True
    assert report.lossy is False
    serialized = report.to_dict()
    assert serialized["lossy"] is False
    assert {column["column_name"] for column in serialized["columns"]} == {"a"}

    _, lossy_report = unify_features(
        [Features({"a": Value("float64")}), Features({"a": Value("int64")})],
        MergeStrategy(allow_lossy=True),
    )
    assert lossy_report.lossy is True
    assert lossy_report.to_dict()["lossy"] is True


def test_class_label_minus_one_missing_label_is_preserved():
    f1 = Features({"label": ClassLabel(names=["a", "b"])})
    f2 = Features({"label": ClassLabel(names=["b", "a"])})
    # -1 encodes an unknown/missing label and must remain -1 after re-indexing.
    d1 = Dataset.from_dict({"label": [0, -1, 1]}, features=f1)
    d2 = Dataset.from_dict({"label": [1, 0]}, features=f2)
    merged = concatenate_datasets([d1, d2])
    assert merged["label"] == [0, -1, 1, 0, 1]
    assert merged["label"][1] == -1
