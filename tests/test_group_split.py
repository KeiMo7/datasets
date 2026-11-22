# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for group-aware (grouped) dataset splitting.

Covers the acceptance criteria of the grouped-split feature: zero shared
groups, ratio deviation reporting, seeded reproducibility (including across
processes and Python hash seeds), pairwise-disjoint 3-way splits, temporal
no-future-sample splits, null group policies, feasibility degradation,
manifest export and state-independent verification.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.features import ClassLabel, Features, Value
from datasets.group_split import (
    GROUP_KEY_RULE,
    GroupSplitError,
    build_groups,
    build_manifest,
    group_key_digest,
    group_split_indices,
    verify_group_split_manifest,
    write_group_split_manifest,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_grouped_columns(n_groups=200, seed=0, n_classes=3, time_localized=False):
    """Rows grouped by customer; each customer appears 1-8 times under
    different row ids, mirroring the leakage scenario from the bug report."""
    rng = np.random.default_rng(seed)
    customers, labels, timestamps = [], [], []
    for cid in range(n_groups):
        k = int(rng.integers(1, 9))
        customers.extend([f"C{cid:04d}"] * k)
        labels.extend([int(rng.integers(0, n_classes))] * k)
        if time_localized:
            # non-overlapping per-group time windows
            base = 10 * cid
            timestamps.extend(
                [np.datetime64("2026-01-01") + np.timedelta64(base + int(rng.integers(0, 8)), "s") for _ in range(k)]
            )
        else:
            timestamps.extend(
                [np.datetime64("2026-01-01") + np.timedelta64(int(rng.integers(0, 1000)), "s") for _ in range(k)]
            )
    return {"customer": customers, "label": labels, "ts": np.asarray(timestamps)}


def _make_dataset(columns):
    ds = Dataset.from_dict(
        {"customer": columns["customer"], "label": columns["label"], "ts": columns["ts"]},
        features=Features(
            {
                "customer": Value("string"),
                "label": ClassLabel(names=[str(i) for i in range(3)]),
                "ts": Value("timestamp[ns]"),
            }
        ),
    )
    return ds


def _owners(columns, indices):
    owner = {}
    for side, rows in indices.items():
        for r in np.asarray(rows).tolist():
            owner.setdefault(columns["customer"][r], set()).add(side)
    return owner


# ---------------------------------------------------------------------------
# Pure-logic tests (no Dataset object)
# ---------------------------------------------------------------------------


class TestBuildGroups:
    def test_groups_collect_all_rows_of_same_key(self):
        columns = {"g": ["a", "b", "a", "a", "b"]}
        groups, dropped, null_rows = build_groups(columns, ["g"])
        assert sorted(groups['["a"]']) == [0, 2, 3]
        assert sorted(groups['["b"]']) == [1, 4]
        assert dropped == [] and null_rows == 0

    def test_multiple_group_columns_composite_key(self):
        columns = {"a": [1, 1, 2], "b": ["x", "x", "x"]}
        groups, _, _ = build_groups(columns, ["a", "b"])
        assert len(groups) == 2
        assert groups['[1,"x"]'] == [0, 1]

    def test_null_policies(self):
        columns = {"g": ["a", None, "b", float("nan"), "a"]}
        with pytest.raises(GroupSplitError):
            build_groups(columns, ["g"], null_group_policy="error")

        groups, dropped, null_rows = build_groups(columns, ["g"], null_group_policy="drop")
        assert null_rows == 2 and dropped == [1, 3]
        assert sorted(groups) == ['["a"]', '["b"]']

        groups, dropped, null_rows = build_groups(columns, ["g"], null_group_policy="singleton")
        assert null_rows == 2 and dropped == []
        # the two null rows must NOT be merged into one shared group
        null_groups = [g for g in groups if g.startswith("__null_singleton__")]
        assert len(null_groups) == 2
        assert all(len(groups[g]) == 1 for g in null_groups)

    def test_unknown_policy_and_missing_column_raise(self):
        with pytest.raises(GroupSplitError):
            build_groups({"g": [1]}, ["g"], null_group_policy="nope")
        with pytest.raises(GroupSplitError):
            build_groups({"g": [1]}, ["missing"])

    def test_group_key_rule_constant_is_canonical(self):
        # the digest rule documented in code must match the implementation
        key = '["a",1]'
        import hashlib

        assert group_key_digest(key) == hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert "sha256" in GROUP_KEY_RULE and "sort_keys=True" in GROUP_KEY_RULE


class TestGroupSplitIndices:
    def test_zero_shared_groups_two_way(self):
        columns = _make_grouped_columns()
        indices, report, _, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=42)
        owner = _owners(columns, indices)
        shared = [k for k, sides in owner.items() if len(sides) > 1]
        assert shared == []
        # every kept row is assigned exactly once
        assert sorted(np.concatenate([indices["train"], indices["test"]]).tolist()) == list(
            range(len(columns["customer"]))
        )

    def test_ratio_deviation_reported_from_actual_assignment(self):
        columns = _make_grouped_columns()
        indices, report, groups, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=42)
        n_kept = sum(len(r) for r in groups.values())
        # recompute actual ratios independently from the produced indices
        actual_test = len(indices["test"]) / n_kept
        assert report.actual_ratios["test"] == pytest.approx(actual_test)
        assert report.ratio_deviations["test"] == pytest.approx(actual_test - 0.25)
        # the report counts must equal the actual index lengths (single source)
        assert report.sample_counts == {s: len(indices[s]) for s in report.side_names}
        assert report.group_counts["train"] + report.group_counts["test"] == len(groups)

    def test_stratified_grouped_split_keeps_label_distribution_close(self):
        columns = _make_grouped_columns()
        _, report, _, _ = group_split_indices(columns, ["customer"], {"test": 0.3}, seed=7, stratify_by_column="label")
        assert report.feasibility == "ok"
        total_per_label = {str(i): sum(1 for v in columns["label"] if v == i) for i in range(3)}
        n_total = len(columns["label"])
        for side in report.side_names:
            side_total = report.sample_counts[side]
            for label, count in report.label_distribution[side].items():
                expected = total_per_label[label] * side_total / n_total
                # whole-group constraint: allow a relaxed, declared tolerance
                assert abs(count - expected) <= 0.12 * side_total

    def test_same_seed_identical(self):
        columns = _make_grouped_columns()
        first, r1, _, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=123)
        second, r2, _, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=123)
        for side in first:
            np.testing.assert_array_equal(first[side], second[side])
        assert r1.to_dict() == r2.to_dict()

    def test_different_seed_keeps_disjointness(self):
        # On uneven group sizes different seeds explore different greedy
        # restarts; regardless of the outcome, group integrity must hold and
        # identical seeds must remain identical (tested separately above).
        rng = np.random.default_rng(0)
        groups = [f"g{i}" for i in range(150) for _ in range(int(rng.integers(1, 20)))]
        columns = {"customer": groups, "label": [0] * len(groups)}
        a, _, _, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=1)
        b, _, _, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=2)
        for indices in (a, b):
            owner = _owners(columns, indices)
            assert all(len(sides) == 1 for sides in owner.values())
        # both runs remain valid whole-group partitions
        assert sorted(np.concatenate([a["train"], a["test"]]).tolist()) == list(range(len(groups)))
        assert sorted(np.concatenate([b["train"], b["test"]]).tolist()) == list(range(len(groups)))

    def test_three_way_pairwise_disjoint(self):
        columns = _make_grouped_columns()
        indices, report, groups, _ = group_split_indices(
            columns, ["customer"], {"validation": 0.1, "test": 0.2}, seed=9
        )
        assert report.side_names == ["train", "validation", "test"]
        owner = _owners(columns, indices)
        assert all(len(sides) == 1 for sides in owner.values())
        side_groups = [{columns["customer"][r] for r in np.asarray(indices[s]).tolist()} for s in report.side_names]
        for i in range(3):
            for j in range(i + 1, 3):
                assert side_groups[i].isdisjoint(side_groups[j])
        assert sum(report.group_counts.values()) == len(groups)
        assert sum(report.sample_counts.values()) == len(columns["customer"])

    def test_temporal_split_has_no_future_sample(self):
        columns = _make_grouped_columns(time_localized=True)
        indices, report, _, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=3, time_column="ts")
        assert report.time_order_ok is True
        assert columns["ts"][indices["train"]].max() <= columns["ts"][indices["test"]].min()

    def test_temporal_three_way_ordering(self):
        columns = _make_grouped_columns(time_localized=True)
        indices, report, _, _ = group_split_indices(
            columns, ["customer"], {"validation": 0.1, "test": 0.2}, seed=4, time_column="ts"
        )
        assert report.time_order_ok is True
        assert columns["ts"][indices["train"]].max() <= columns["ts"][indices["validation"]].min()
        assert columns["ts"][indices["validation"]].max() <= columns["ts"][indices["test"]].min()

    def test_temporal_infeasibility_raises(self):
        # groups spanning the whole timeline: no ordered cut can satisfy the
        # row-level no-future constraint -> explicit failure, not a bad split
        columns = _make_grouped_columns(time_localized=False)
        with pytest.raises(GroupSplitError, match="No temporal cut"):
            group_split_indices(columns, ["customer"], {"test": 0.25}, seed=1, time_column="ts")

    def test_imbalanced_groups_degrade_with_measured_deviation(self):
        columns = {"g": ["X"] * 500 + [f"g{i}" for i in range(100)], "l": [0] * 600}
        _, report, _, _ = group_split_indices(
            columns, ["g"], {"test": 0.25}, seed=5, stratify_by_column="l", ratio_tolerance=0.02
        )
        assert report.feasibility == "degraded"
        assert any("Largest group" in w for w in report.warnings)
        # deviation numbers are real and bounded
        assert abs(report.ratio_deviations["test"]) > 0.02
        with pytest.raises(GroupSplitError):
            group_split_indices(
                columns,
                ["g"],
                {"test": 0.25},
                seed=5,
                stratify_by_column="l",
                ratio_tolerance=0.02,
                strict=True,
            )

    def test_integer_sizes_supported(self):
        columns = _make_grouped_columns(n_groups=50)
        indices, report, _, _ = group_split_indices(columns, ["customer"], {"test": 20}, seed=0)
        owner = _owners(columns, indices)
        assert all(len(s) == 1 for s in owner.values())
        assert report.sample_counts["test"] >= 20  # whole groups only

    def test_too_few_groups_raise(self):
        columns = {"g": ["a", "a", "b", "b"], "l": [0, 0, 1, 1]}
        with pytest.raises(GroupSplitError, match="at least 3 groups"):
            group_split_indices(columns, ["g"], {"validation": 0.1, "test": 0.2}, seed=0)

    def test_seed_required_for_reproducibility(self):
        columns = _make_grouped_columns(n_groups=20)
        with pytest.raises(GroupSplitError):
            group_split_indices(columns, ["customer"], {"test": 0.25}, seed=None)


# ---------------------------------------------------------------------------
# Manifest + independent verification
# ---------------------------------------------------------------------------


def _run_split_and_manifest(tmp_path, columns, **kwargs):
    indices, report, groups, dropped = group_split_indices(columns, ["customer"], seed=11, **kwargs)
    assignment = {
        side: [g for g, gr in groups.items() if gr[0] in set(np.asarray(indices[side]).tolist())]
        for side in report.side_names
    }
    manifest = build_manifest(groups, assignment, report, dropped)
    path = os.path.join(tmp_path, "manifest.json")
    write_group_split_manifest(manifest, path)
    return path, indices, report


class TestManifest:
    def test_manifest_roundtrip_and_verification(self, tmp_path):
        columns = _make_grouped_columns()
        path, _, report = _run_split_and_manifest(
            tmp_path, columns, side_sizes={"test": 0.3}, stratify_by_column="label"
        )
        result = verify_group_split_manifest(path)
        assert result["ok"] is True
        assert result["shared_groups"] == []
        with open(path) as f:
            manifest = json.load(f)
        assert manifest["group_key_rule"] == GROUP_KEY_RULE
        assert manifest["report"]["sample_counts"] == report.sample_counts

    def test_verification_against_raw_columns(self, tmp_path):
        columns = _make_grouped_columns()
        path, _, _ = _run_split_and_manifest(tmp_path, columns, side_sizes={"test": 0.3})
        result = verify_group_split_manifest(path, columns=columns)
        assert result["ok"] is True

    def test_verification_detects_cross_side_tampering(self, tmp_path):
        columns = _make_grouped_columns()
        path, _, _ = _run_split_and_manifest(tmp_path, columns, side_sizes={"test": 0.3})
        with open(path) as f:
            manifest = json.load(f)
        target = next(row for row in manifest["rows"] if row["side"] == "test")
        target["side"] = "train"
        bad_path = os.path.join(tmp_path, "bad.json")
        write_group_split_manifest(manifest, bad_path)
        result = verify_group_split_manifest(bad_path, columns=columns)
        assert result["ok"] is False
        assert any("different from their group" in e for e in result["errors"])

    def test_verification_detects_digest_tampering(self, tmp_path):
        columns = _make_grouped_columns()
        path, _, _ = _run_split_and_manifest(tmp_path, columns, side_sizes={"test": 0.3})
        with open(path) as f:
            manifest = json.load(f)
        manifest["rows"][0]["group_key_digest"] = "0" * 64
        bad_path = os.path.join(tmp_path, "bad2.json")
        write_group_split_manifest(manifest, bad_path)
        result = verify_group_split_manifest(bad_path, columns=columns)
        assert result["ok"] is False

    def test_atomic_write_no_half_manifest_and_residue_cleanup(self, tmp_path):
        columns = _make_grouped_columns(n_groups=30)
        path, _, _ = _run_split_and_manifest(tmp_path, columns, side_sizes={"test": 0.25})
        # simulate a crashed previous run leaving a temp file
        with open(path + ".tmp", "w") as f:
            f.write("partial")
        indices, report, groups, _ = group_split_indices(columns, ["customer"], {"test": 0.25}, seed=11)
        assignment = {
            side: [g for g, gr in groups.items() if gr[0] in set(np.asarray(indices[side]).tolist())]
            for side in report.side_names
        }
        write_group_split_manifest(build_manifest(groups, assignment, report), path)
        assert not os.path.exists(path + ".tmp")
        assert verify_group_split_manifest(path)["ok"] is True

    def test_manifest_covers_every_row_with_digest(self, tmp_path):
        columns = _make_grouped_columns()
        path, _, _ = _run_split_and_manifest(tmp_path, columns, side_sizes={"test": 0.25})
        with open(path) as f:
            manifest = json.load(f)
        row_indices = sorted(row["row_index"] for row in manifest["rows"])
        assert row_indices == list(range(len(columns["customer"])))
        assert all(row["group_key_digest"] for row in manifest["rows"])


# ---------------------------------------------------------------------------
# Dataset integration tests
# ---------------------------------------------------------------------------


class TestDatasetGroupSplit:
    def test_group_train_test_split_zero_shared_groups(self):
        ds = _make_dataset(_make_grouped_columns())
        out = ds.group_train_test_split(group_by_columns="customer", test_size=0.25, seed=42)
        assert isinstance(out, DatasetDict)
        assert set(out) == {"train", "test"}
        train_groups = set(out["train"]["customer"])
        test_groups = set(out["test"]["customer"])
        assert train_groups.isdisjoint(test_groups)
        assert len(out["train"]) + len(out["test"]) == len(ds)
        # report and products share one source
        assert out.split_report.sample_counts == {"train": len(out["train"]), "test": len(out["test"])}

    def test_group_train_test_split_three_way(self):
        ds = _make_dataset(_make_grouped_columns())
        out = ds.group_train_test_split(group_by_columns=["customer"], test_size=0.2, val_size=0.1, seed=8)
        assert set(out) == {"train", "validation", "test"}
        groups_per_side = [set(out[s]["customer"]) for s in ("train", "validation", "test")]
        for i in range(3):
            for j in range(i + 1, 3):
                assert groups_per_side[i].isdisjoint(groups_per_side[j])

    def test_group_train_test_split_time_constraint(self):
        ds = _make_dataset(_make_grouped_columns(time_localized=True))
        out = ds.group_train_test_split(group_by_columns="customer", test_size=0.25, seed=2, time_column="ts")
        assert out.split_report.time_order_ok is True
        assert max(out["train"]["ts"]) <= min(out["test"]["ts"])

    def test_manifest_export_and_dataset_verification(self, tmp_path):
        ds = _make_dataset(_make_grouped_columns())
        manifest_path = os.path.join(tmp_path, "split.json")
        out = ds.group_train_test_split(
            group_by_columns="customer", test_size=0.3, seed=5, manifest_path=manifest_path
        )
        assert os.path.exists(manifest_path)
        assert out.split_manifest_path == os.path.abspath(manifest_path)
        result = ds.verify_group_split(manifest_path)
        assert result["ok"] is True and result["shared_groups"] == []

    def test_null_group_policies_on_dataset(self):
        ds = _make_dataset(_make_grouped_columns())
        table = ds.to_dict()
        table["customer"][0] = None
        table["customer"][3] = None
        ds_null = Dataset.from_dict(table, features=ds.features)
        with pytest.raises(GroupSplitError):
            ds_null.group_train_test_split(group_by_columns="customer", test_size=0.25, seed=0)

        out_drop = ds_null.group_train_test_split(
            group_by_columns="customer", test_size=0.25, seed=0, null_group_policy="drop"
        )
        assert out_drop.split_report.null_rows == 2
        assert sum(out_drop.split_report.sample_counts.values()) == len(ds) - 2

        out_single = ds_null.group_train_test_split(
            group_by_columns="customer", test_size=0.25, seed=0, null_group_policy="singleton"
        )
        assert sum(out_single.split_report.sample_counts.values()) == len(ds)

    def test_dataset_structure_and_features_unchanged(self):
        ds = _make_dataset(_make_grouped_columns())
        original_features = ds.features
        out = ds.group_train_test_split(group_by_columns="customer", test_size=0.25, seed=0)
        assert out["train"].features == original_features
        assert out["test"].features == original_features
        # original dataset untouched
        assert len(ds) == sum(out.split_report.sample_counts.values())

    def test_rejects_indexed_dataset(self):
        from unittest.mock import patch

        ds = _make_dataset(_make_grouped_columns())
        with patch.object(Dataset, "list_indexes", return_value=["search_index"]):
            with pytest.raises(Exception):
                ds.group_train_test_split(group_by_columns="customer", test_size=0.25, seed=0)


# ---------------------------------------------------------------------------
# Cross-process reproducibility (acceptance criterion n°3 / hard constraint n°1)
# ---------------------------------------------------------------------------


_CHILD_SCRIPT = """
import hashlib, json, sys
sys.path.insert(0, {src!r})
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel
rng = np.random.default_rng(7)
customers, labels = [], []
for cid in range(120):
    k = int(rng.integers(1, 8))
    customers.extend([f"C{{cid:04d}}"] * k)
    labels.extend([int(rng.integers(0, 3))] * k)
ds = Dataset.from_dict({{"customer": customers, "label": labels}}, features=Features({{
    "customer": Value("string"), "label": ClassLabel(names=["0","1","2"])}}))
out = ds.group_train_test_split(group_by_columns="customer", test_size=0.25, seed=2024)
sig = {{s: hashlib.sha256("\\u0001".join(list(out[s]["customer"])).encode()).hexdigest()
       for s in out}}
print(json.dumps(sig, sort_keys=True))
"""


def test_cross_process_reproducible_across_hash_seeds(tmp_path):
    import datasets

    src = os.path.dirname(os.path.dirname(datasets.__file__))
    script = tmp_path / "child.py"
    script.write_text(_CHILD_SCRIPT.format(src=src))
    signatures = []
    for hash_seed in ("0", "12345", "999"):
        env = dict(os.environ, PYTHONPATH=src, PYTHONHASHSEED=hash_seed)
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, check=True)
        signatures.append(proc.stdout.strip())
    assert signatures[0] == signatures[1] == signatures[2]
    assert len(signatures[0]) > 0
