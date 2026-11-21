import json

import pytest

import datasets
from datasets import Dataset, DatasetDict, SplitIntegrityConfig
from datasets.exceptions import (
    FingerprintMismatchError,
    MissingSplitError,
    SplitOverlapError,
)
from datasets.split_integrity import (
    OverlapReport,
    SplitManifest,
    assign_splits,
    build_split_manifest,
    detect_split_overlap,
    split_by_seed,
)


def _dataset_three_splits(overlap: bool = False):
    if overlap:
        return DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"id": list(range(80)), "text": [f"train-{i}" for i in range(80)]}, split="train"
                ),
                "validation": Dataset.from_dict(
                    {"id": list(range(60, 70)), "text": [f"val-{i}" for i in range(60, 70)]},
                    split="validation",
                ),
                "test": Dataset.from_dict(
                    {"id": [0, 1, 80, 81], "text": ["test-0", "test-1", "test-80", "test-81"]}, split="test"
                ),
            }
        )
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {"id": list(range(80)), "text": [f"train-{i}" for i in range(80)]}, split="train"
            ),
            "validation": Dataset.from_dict(
                {"id": list(range(80, 90)), "text": [f"val-{i}" for i in range(80, 90)]}, split="validation"
            ),
            "test": Dataset.from_dict(
                {"id": list(range(90, 110)), "text": [f"test-{i}" for i in range(90, 110)]}, split="test"
            ),
        }
    )


# --------------------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------------------


def test_content_fingerprint_depends_on_values_and_order_not_files(tmp_path):
    ds = Dataset.from_dict(
        {"id": list(range(30)), "text": [f"row-{i}" for i in range(30)]}, split="train"
    )
    shuffled = Dataset.from_dict(
        {"id": list(reversed(range(30))), "text": [f"row-{i}" for i in reversed(range(30))]}, split="train"
    )
    changed = Dataset.from_dict(
        {"id": list(range(30)), "text": [f"other-{i}" for i in range(30)]}, split="train"
    )

    assert ds.content_fingerprint() == ds.content_fingerprint()
    assert ds.content_fingerprint() != shuffled.content_fingerprint()
    assert ds.content_fingerprint() != changed.content_fingerprint()

    # Identical content saved with a different number of shards has the same
    # fingerprint: shard layout, file names and mtimes don't participate.
    ds.save_to_disk(tmp_path / "one", num_shards=1)
    ds.save_to_disk(tmp_path / "four", num_shards=4)
    loaded_one = Dataset.load_from_disk(tmp_path / "one")
    loaded_four = Dataset.load_from_disk(tmp_path / "four")
    assert len(loaded_one.cache_files) == 1
    assert len(loaded_four.cache_files) == 4
    assert loaded_one.content_fingerprint() == loaded_four.content_fingerprint() == ds.content_fingerprint()


def test_content_fingerprint_is_independent_of_batch_size_and_num_proc():
    ds = Dataset.from_dict({"id": list(range(137)), "text": [str(i) for i in range(137)]}, split="train")
    assert ds.content_fingerprint(batch_size=1) == ds.content_fingerprint(batch_size=1000)
    assert ds.content_fingerprint(batch_size=7, num_proc=2) == ds.content_fingerprint(batch_size=7)


def test_content_fingerprint_honours_selection_order():
    ds = Dataset.from_dict({"id": [1, 2, 3]}, split="train")
    assert ds.select([2, 0]).content_fingerprint() == Dataset.from_dict(
        {"id": [3, 1]}, split="train"
    ).content_fingerprint()
    assert ds.select([2, 0]).content_fingerprint() != ds.select([0, 2]).content_fingerprint()


# --------------------------------------------------------------------------------------
# Normal load with fingerprint
# --------------------------------------------------------------------------------------


def test_save_and_load_from_disk_with_integrity(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path, split_integrity={"seed": 42, "split_method": "seeded"})

    manifest_raw = json.loads((path / "split_integrity.json").read_text())
    assert manifest_raw["split_method"] == "seeded"
    assert manifest_raw["seed"] == 42
    assert manifest_raw["num_samples"] == {"train": 80, "validation": 10, "test": 20}
    for split_name in dataset_dict:
        assert manifest_raw["fingerprints"][split_name] == dataset_dict[split_name].content_fingerprint()

    loaded = datasets.load_from_disk(path, split_integrity=True)
    # Existing return structure is unchanged.
    assert isinstance(loaded, DatasetDict)
    assert list(loaded) == ["train", "validation", "test"]
    for split_name in loaded:
        assert loaded[split_name].to_dict() == dataset_dict[split_name].to_dict()
        # Each split records its provenance without changing its public API.
        metadata = loaded[split_name]._split_integrity
        assert metadata["split_method"] == "seeded"
        assert metadata["fingerprint"] == dataset_dict[split_name].content_fingerprint()

    assert loaded.verify_splits().fingerprints == manifest_raw["fingerprints"]


def test_load_without_integrity_unchanged_when_flag_is_off(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path, split_integrity=True)

    # Without the new keyword, no manifest is looked at and the result is identical.
    loaded = datasets.load_from_disk(path)
    for split_name in dataset_dict:
        assert loaded[split_name].to_dict() == dataset_dict[split_name].to_dict()
        assert not hasattr(loaded[split_name], "_split_integrity")

    # With verification off, an unreadable cache file raises the same pyarrow error as
    # before this feature existed, not a FingerprintMismatchError.
    arrow = next((path / "train").glob("*.arrow"))
    original = arrow.read_bytes()
    arrow.write_bytes(b"not an arrow file")
    with pytest.raises(Exception) as excinfo:
        datasets.load_from_disk(path)
    assert not isinstance(excinfo.value, FingerprintMismatchError)
    arrow.write_bytes(original)


def test_strict_missing_manifest(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path)  # no manifest

    with pytest.raises(datasets.exceptions.SplitManifestError):
        datasets.load_from_disk(path, split_integrity=True)

    # Opt out explicitly: behaves exactly like a plain load.
    loaded = datasets.load_from_disk(path, split_integrity={"strict_missing_manifest": False})
    for split_name in dataset_dict:
        assert loaded[split_name].to_dict() == dataset_dict[split_name].to_dict()


# --------------------------------------------------------------------------------------
# Overlap detection
# --------------------------------------------------------------------------------------


def test_overlap_detection_reports_sample_ids():
    dataset_dict = _dataset_three_splits(overlap=True)
    report = detect_split_overlap(dataset_dict, id_columns="id")
    assert isinstance(report, OverlapReport)
    assert report.has_overlap
    assert report.total_overlapping_unique_ids == 12  # ids 0,1 and 60..69
    assert report.overlap_pairs["test__train"]["sample_ids"] == [0, 1]
    assert report.overlap_pairs["test__train"]["num_overlapping_ids"] == 2
    assert report.overlap_pairs["test__train"]["num_rows"] == 4
    assert "train__validation" in report.overlap_pairs  # ids 60..69 are in both
    assert report.overlap_pairs["train__validation"]["num_overlapping_ids"] == 10

    with pytest.raises(SplitOverlapError) as excinfo:
        dataset_dict.check_overlap(id_columns="id")
    assert excinfo.value.report.total_overlapping_unique_ids == 12
    assert "0" in str(excinfo.value) and "1" in str(excinfo.value)


def test_overlap_is_by_identifier_not_content():
    # Same content, different identifiers: NOT an overlap.
    same_content = DatasetDict(
        {
            "train": Dataset.from_dict({"id": [1], "text": ["hello"]}, split="train"),
            "test": Dataset.from_dict({"id": [2], "text": ["hello"]}, split="test"),
        }
    )
    assert not detect_split_overlap(same_content, id_columns="id").has_overlap

    # Different content, same identifier: IS an overlap.
    same_id = DatasetDict(
        {
            "train": Dataset.from_dict({"id": [1], "text": ["hello"]}, split="train"),
            "test": Dataset.from_dict({"id": [1], "text": ["goodbye"]}, split="test"),
        }
    )
    report = detect_split_overlap(same_id, id_columns="id")
    assert report.has_overlap
    assert report.overlap_pairs["test__train"]["sample_ids"] == [1]
    # The report says *which* identifiers leaked; where_is_sample says exactly where.
    assert datasets.where_is_sample(same_id, 1) == {"train": [0], "test": [0]}
    assert datasets.where_is_sample(same_id, 2) == {}


def test_identifier_normalisation_across_numeric_types():
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"id": [1, 2.0]}, split="train"),
            "test": Dataset.from_dict({"id": [1.0, 2]}, split="test"),
        }
    )
    report = detect_split_overlap(dataset_dict, id_columns="id")
    assert report.total_overlapping_unique_ids == 2


def test_overlap_duplicate_within_split_is_counted():
    dataset_dict = DatasetDict(
        {"train": Dataset.from_dict({"id": [1, 1, 2], "text": ["a", "b", "c"]}, split="train")}
    )
    report = detect_split_overlap(dataset_dict, id_columns="id")
    assert not report.has_overlap
    assert report.duplicate_ids_within_split["train"] == 1
    assert report.num_unique_ids["train"] == 2


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 1000])
def test_overlap_chunk_size_does_not_change_conclusion(chunk_size):
    dataset_dict = _dataset_three_splits(overlap=True)
    report = detect_split_overlap(dataset_dict, id_columns="id", chunk_size=chunk_size)
    baseline = detect_split_overlap(dataset_dict, id_columns="id", chunk_size=1000)
    assert report.to_dict() | {"chunk_size": None} == baseline.to_dict() | {"chunk_size": None}
    assert report.overlap_pairs == baseline.overlap_pairs
    assert report.num_unique_ids == baseline.num_unique_ids
    assert report.duplicate_ids_within_split == baseline.duplicate_ids_within_split
    assert report.total_overlapping_unique_ids == baseline.total_overlapping_unique_ids


def test_overlap_across_chunk_boundary_is_not_missed():
    # id 5 is the last row of chunk 1 and the first row of chunk 2 when chunk_size=3.
    left = Dataset.from_dict({"id": [1, 2, 5]}, split="train")
    right = Dataset.from_dict({"id": [5, 6, 7]}, split="test")
    dataset_dict = DatasetDict({"train": left, "test": right})
    for chunk_size in (1, 2, 3, 5, 6, 100):
        report = detect_split_overlap(dataset_dict, id_columns="id", chunk_size=chunk_size)
        assert report.has_overlap, chunk_size
        assert report.overlap_pairs["test__train"]["sample_ids"] == [5]
        assert report.overlap_pairs["test__train"]["num_rows"] == 2


def test_overlap_truncates_large_reports():
    ids = list(range(1000))
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"id": ids}, split="train"),
            "test": Dataset.from_dict({"id": ids}, split="test"),
        }
    )
    report = detect_split_overlap(dataset_dict, id_columns="id", max_reported_overlaps=5)
    assert report.total_overlapping_unique_ids == 1000
    assert len(report.overlap_pairs["test__train"]["sample_ids"]) == 5
    assert report.truncated
    # The summary stays short: counts plus a few examples, never 1000 lines.
    assert len(report.summary().splitlines()) == 2


def test_missing_split_for_overlap_is_named():
    dataset_dict = DatasetDict({"train": Dataset.from_dict({"id": [1]}, split="train")})
    with pytest.raises(MissingSplitError, match="validation"):
        dataset_dict.check_overlap(id_columns="id", split_names=["train", "validation"])


# --------------------------------------------------------------------------------------
# Seeded replay
# --------------------------------------------------------------------------------------


def test_same_seed_splits_are_identical_row_by_row():
    ds = Dataset.from_dict({"id": list(range(500)), "text": [str(i) for i in range(500)]})

    first = split_by_seed(ds, seed=1234)
    second = split_by_seed(ds, seed=1234)
    assert list(first) == list(second) == ["train", "validation", "test"]
    for split_name in first:
        # Row-by-row identity, including order.
        assert first[split_name]["id"] == second[split_name]["id"]
        assert first[split_name].to_dict() == second[split_name].to_dict()

    other_seed = split_by_seed(ds, seed=5678)
    assert any(
        first[split_name]["id"] != other_seed[split_name]["id"] for split_name in ("train", "validation", "test")
    )


def test_seeded_split_is_independent_of_row_order_and_sharding():
    ids = list(range(200))
    ds = Dataset.from_dict({"id": ids, "text": [str(i) for i in ids]})
    assignment = assign_splits(ds, seed=7)
    # selection / reordering must not move a sample to another split
    reordered = ds.select(list(reversed(range(len(ds)))))
    reordered_assignment = assign_splits(reordered, seed=7)
    for split_name in assignment.assignments:
        assigned_ids = {ids[i] for i in assignment.assignments[split_name]}
        assigned_ids_reordered = {
            reordered["id"][i] for i in reordered_assignment.assignments[split_name]
        }
        assert assigned_ids == assigned_ids_reordered

    # a subset keeps the same assignment for the samples it contains
    subset = ds.select(range(0, 200, 2))
    subset_assignment = assign_splits(subset, seed=7)
    for split_name in assignment.assignments:
        full_ids = {ids[i] for i in assignment.assignments[split_name]}
        subset_ids = {subset["id"][i] for i in subset_assignment.assignments[split_name]}
        assert subset_ids == full_ids & set(subset["id"])


def test_seeded_split_has_no_overlap_and_covers_every_row():
    ds = Dataset.from_dict({"id": list(range(1000))})
    splits = split_by_seed(ds, seed=42, ratios={"train": 0.8, "validation": 0.1, "test": 0.1})
    report = detect_split_overlap(splits, id_columns="id")
    assert not report.has_overlap
    all_ids = sorted(sum((list(splits[name]["id"]) for name in splits), []))
    assert all_ids == list(range(1000))


def test_seeded_manifest_is_replayable(tmp_path):
    ds = Dataset.from_dict({"id": list(range(300)), "text": [str(i) for i in range(300)]})
    splits = split_by_seed(ds, seed=2024)
    manifest = build_split_manifest(splits, split_method="seeded", seed=2024, include_sample_ids=True)
    path = manifest.save(tmp_path / "manifest.json")

    # Someone else replays the seed and compares against the committed manifest.
    replayed = split_by_seed(ds, seed=2024)
    replay_manifest = build_split_manifest(replayed, split_method="seeded", seed=2024, include_sample_ids=True)
    loaded_manifest = SplitManifest.load(path)
    assert loaded_manifest.sample_ids == replay_manifest.sample_ids
    assert loaded_manifest.fingerprints == replay_manifest.fingerprints
    loaded_manifest.verify(replayed)


# --------------------------------------------------------------------------------------
# Cache replacement / corruption
# --------------------------------------------------------------------------------------


def test_replaced_cache_file_is_rejected(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path, split_integrity=True)

    # Replace the train split with valid Arrow data that doesn't match the manifest.
    replacement = Dataset.from_dict({"id": [0], "text": ["replaced"]}, split="train")
    replacement.save_to_disk(path / "train")

    with pytest.raises(FingerprintMismatchError) as excinfo:
        datasets.load_from_disk(path, split_integrity=True)
    message = str(excinfo.value)
    assert "train" in message
    assert "fingerprint" in message.lower()
    assert "data-00000-of-00001.arrow" in message


def test_corrupted_cache_file_is_rejected(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path, split_integrity=True)

    arrow = next((path / "test").glob("*.arrow"))
    arrow.write_bytes(b"not an arrow file")

    with pytest.raises(FingerprintMismatchError) as excinfo:
        datasets.load_from_disk(path, split_integrity=True)
    assert "test" in str(excinfo.value)
    assert "data-00000-of-00001.arrow" in str(excinfo.value)


def test_manifest_missing_split_is_named(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path, split_integrity=True)

    manifest = SplitManifest.load(path / "split_integrity.json")
    loaded = {name: dataset for name, dataset in dataset_dict.items() if name != "validation"}
    with pytest.raises(MissingSplitError, match="validation"):
        manifest.verify(loaded)


# --------------------------------------------------------------------------------------
# load_dataset opt-in (csv builder, no network)
# --------------------------------------------------------------------------------------

csv = pytest.importorskip("csv")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        writer.writerows(rows)


def test_load_dataset_records_fingerprints_and_manifest(tmp_path):
    _write_csv(tmp_path / "train.csv", [(i, f"train-{i}") for i in range(20)])
    _write_csv(tmp_path / "test.csv", [(i, f"test-{i}") for i in range(20, 25)])
    data_files = {"train": str(tmp_path / "train.csv"), "test": str(tmp_path / "test.csv")}
    export = tmp_path / "exported_manifest.json"

    ds = datasets.load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=str(tmp_path / "cache"),
        split_integrity={"manifest_path": str(export)},
    )
    assert list(ds) == ["train", "test"]
    manifest = json.loads(export.read_text())
    assert manifest["num_samples"] == {"train": 20, "test": 5}
    assert manifest["fingerprints"]["train"] == ds["train"].content_fingerprint()
    for split_name in ds:
        assert ds[split_name]._split_integrity["fingerprint"] == manifest["fingerprints"][split_name]

    # The manifest was also written next to the builder cache.
    cache_manifests = list((tmp_path / "cache").rglob("split_integrity.json"))
    assert cache_manifests

    # A plain load is byte-for-byte identical and has no extra attributes.
    plain = datasets.load_dataset("csv", data_files=data_files, cache_dir=str(tmp_path / "cache"))
    for split_name in plain:
        assert plain[split_name].to_dict() == ds[split_name].to_dict()
        assert not hasattr(plain[split_name], "_split_integrity")


def test_load_dataset_refuses_tampered_builder_cache_manifest(tmp_path):
    _write_csv(tmp_path / "train.csv", [(i, f"train-{i}") for i in range(20)])
    cache = tmp_path / "cache"
    datasets.load_dataset(
        "csv", data_files=str(tmp_path / "train.csv"), cache_dir=str(cache), split_integrity=True
    )
    manifest_path = next(cache.rglob("split_integrity.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprints"]["train"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(FingerprintMismatchError) as excinfo:
        datasets.load_dataset(
            "csv", data_files=str(tmp_path / "train.csv"), cache_dir=str(cache), split_integrity=True
        )
    assert "train" in str(excinfo.value)

    # Without the flag the tampered manifest is irrelevant.
    assert len(datasets.load_dataset("csv", data_files=str(tmp_path / "train.csv"), cache_dir=str(cache))["train"]) == 20


def test_load_dataset_detects_overlap(tmp_path):
    _write_csv(tmp_path / "train.csv", [(i, f"train-{i}") for i in range(10)])
    _write_csv(tmp_path / "test.csv", [(i, f"test-{i}") for i in range(5, 8)])
    data_files = {"train": str(tmp_path / "train.csv"), "test": str(tmp_path / "test.csv")}

    with pytest.raises(SplitOverlapError) as excinfo:
        datasets.load_dataset(
            "csv", data_files=data_files, cache_dir=str(tmp_path / "cache"), split_integrity=True
        )
    report = excinfo.value.report
    assert report.overlap_pairs["test__train"]["sample_ids"] == [5, 6, 7]


def test_split_integrity_config_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown_option"):
        SplitIntegrityConfig.coerce({"unknown_option": 1})
    assert not SplitIntegrityConfig.coerce(None).enabled
    assert SplitIntegrityConfig.coerce(True).enabled
    assert SplitIntegrityConfig(id_columns=["a", "b"]).normalized_id_columns() == ["a", "b"]


def test_splits_without_identifier_still_get_fingerprints(tmp_path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["a", "b"]}, split="train"),
            "test": Dataset.from_dict({"text": ["c"]}, split="test"),
        }
    )
    path = tmp_path / "no_id"
    dataset_dict.save_to_disk(path, split_integrity=True)
    manifest = json.loads((path / "split_integrity.json").read_text())
    assert manifest["sample_ids"] == {"train": None, "test": None}
    assert manifest["fingerprints"]["train"] == dataset_dict["train"].content_fingerprint()

    loaded = datasets.load_from_disk(path, split_integrity=True)
    # Overlap detection needs an identifier, so it says so instead of guessing.
    with pytest.raises(MissingSplitError, match="identifier column"):
        loaded.verify_splits()
    assert loaded.verify_splits(config={"require_id_column": False}).num_samples == {"train": 2, "test": 1}


def test_required_splits_are_named_on_load_dataset(tmp_path):
    _write_csv(tmp_path / "train.csv", [(i, f"train-{i}") for i in range(5)])
    with pytest.raises(MissingSplitError, match="validation"):
        datasets.load_dataset(
            "csv",
            data_files={"train": str(tmp_path / "train.csv")},
            cache_dir=str(tmp_path / "cache"),
            split_integrity={"required_splits": ["train", "validation"]},
        )


def test_streaming_with_integrity_is_rejected(tmp_path):
    _write_csv(tmp_path / "train.csv", [(i, f"train-{i}") for i in range(5)])
    with pytest.raises(ValueError, match="materialised splits"):
        datasets.load_dataset(
            "csv",
            data_files={"train": str(tmp_path / "train.csv")},
            streaming=True,
            split_integrity=True,
        )


def test_malformed_or_future_manifest_is_rejected(tmp_path):
    dataset_dict = _dataset_three_splits()
    path = tmp_path / "dataset"
    dataset_dict.save_to_disk(path, split_integrity=True)
    manifest_path = path / "split_integrity.json"

    raw = json.loads(manifest_path.read_text())
    raw["future_field"] = 1
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(datasets.exceptions.SplitManifestError, match="future_field"):
        datasets.load_from_disk(path, split_integrity=True)

    raw.pop("future_field")
    raw["split_integrity_version"] = 99
    manifest_path.write_text(json.dumps(raw))
    with pytest.raises(datasets.exceptions.SplitManifestError, match="version"):
        datasets.load_from_disk(path, split_integrity=True)
