#!/usr/bin/env python3
"""Acceptance demo for the group-aware split feature (UID 75).

Reproduces the leakage scenario from the bug report (the same customer appears
under different sample ids / collection times) and demonstrates every
acceptance criterion:

1. zero shared groups between train and evaluation sides;
2. ratio deviation measured against the actual products;
3. seeded reproducibility;
4. pairwise-disjoint 3-way split and temporal no-future-sample split;
5. manifest export and state-independent verification.

Run from the repository root:

    PYTHONPATH=src python3 -P output/acceptance_demo.py

All products are written under output/.
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from datasets import Dataset, verify_group_split_manifest  # noqa: E402
from datasets.features import ClassLabel, Features, Value  # noqa: E402
from datasets.group_split import GROUP_KEY_RULE  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
TOLERANCE = 0.05


def build_leakage_scenario():
    """Same customer, multiple rows with distinct sample ids and timestamps."""
    rng = np.random.default_rng(2026)
    rows = []
    sample_id = 0
    for customer_id in range(200):
        n_rows = int(rng.integers(1, 9))
        # time-localized windows so a temporal cut is feasible
        window_start = 10 * customer_id
        for _ in range(n_rows):
            rows.append(
                {
                    "sample_id": f"S{sample_id:06d}",
                    "customer_id": f"C{customer_id:04d}",
                    "label": int(rng.integers(0, 3)),
                    "collected_at": np.datetime64("2026-01-01")
                    + np.timedelta64(window_start + int(rng.integers(0, 8)), "s"),
                }
            )
            sample_id += 1
    features = Features(
        {
            "sample_id": Value("string"),
            "customer_id": Value("string"),
            "label": ClassLabel(names=["a", "b", "c"]),
            "collected_at": Value("timestamp[ns]"),
        }
    )
    return Dataset.from_list(rows, features=features)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ds = build_leakage_scenario()
    print(f"Synthetic leakage scenario: {len(ds)} rows, {len(set(ds['customer_id']))} customer groups")

    # ------------------------------------------------------------------ 1/2
    section("1. Two-way grouped + stratified split: zero shared groups, ratio deviation")
    manifest_2way = os.path.join(OUTPUT_DIR, "manifest_2way.json")
    out = ds.group_train_test_split(
        group_by_columns="customer_id",
        test_size=0.25,
        seed=42,
        stratify_by_column="label",
        manifest_path=manifest_2way,
    )
    report = out.split_report
    train_groups = set(out["train"]["customer_id"])
    test_groups = set(out["test"]["customer_id"])
    print(f"shared groups train/test            : {len(train_groups & test_groups)}")
    print(f"sample counts                       : {report.sample_counts}")
    print(f"group counts                        : {report.group_counts}")
    print(f"target ratios                       : { {k: round(v, 4) for k, v in report.target_ratios.items()} }")
    print(f"actual ratios                       : { {k: round(v, 4) for k, v in report.actual_ratios.items()} }")
    print(f"ratio deviations                    : { {k: round(v, 6) for k, v in report.ratio_deviations.items()} }")
    print(f"within tolerance {TOLERANCE}             : {all(abs(d) <= TOLERANCE for d in report.ratio_deviations.values())}")
    print(f"feasibility                         : {report.feasibility}")
    print(f"label distribution                  : {report.label_distribution}")

    # ------------------------------------------------------------------ 3
    section("2. Seeded reproducibility (same seed, two independent calls)")
    out_a = ds.group_train_test_split(group_by_columns="customer_id", test_size=0.25, seed=2024)
    out_b = ds.group_train_test_split(group_by_columns="customer_id", test_size=0.25, seed=2024)
    identical = (
        list(out_a["train"]["sample_id"]) == list(out_b["train"]["sample_id"])
        and list(out_a["test"]["sample_id"]) == list(out_b["test"]["sample_id"])
    )
    print(f"identical assignments with same seed: {identical}")

    # ------------------------------------------------------------------ 4
    section("3. Three-way split: pairwise group-disjoint")
    manifest_3way = os.path.join(OUTPUT_DIR, "manifest_3way.json")
    out3 = ds.group_train_test_split(
        group_by_columns=["customer_id"],
        test_size=0.2,
        val_size=0.1,
        seed=7,
        manifest_path=manifest_3way,
    )
    side_groups = {side: set(out3[side]["customer_id"]) for side in ("train", "validation", "test")}
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    for a, b in pairs:
        print(f"shared groups {a:<10}/{b:<10}: {len(side_groups[a] & side_groups[b])}")
    print(f"sample counts                       : {out3.split_report.sample_counts}")
    print(f"group counts                        : {out3.split_report.group_counts}")

    # ------------------------------------------------------------------ 4b
    section("4. Temporal split: no future sample in train")
    outt = ds.group_train_test_split(
        group_by_columns="customer_id", test_size=0.25, seed=3, time_column="collected_at"
    )
    train_max = max(outt["train"]["collected_at"])
    test_min = min(outt["test"]["collected_at"])
    print(f"max(train time) <= min(test time)   : {train_max <= test_min}")
    print(f"report.time_order_ok                : {outt.split_report.time_order_ok}")
    print(f"ratio deviations                    : "
          f"{ {k: round(v, 6) for k, v in outt.split_report.ratio_deviations.items()} }")

    # ------------------------------------------------------------------ 5
    section("5. Manifest export and state-independent verification")
    for path in (manifest_2way, manifest_3way):
        file_only = verify_group_split_manifest(path)
        data_backed = verify_group_split_manifest(
            path, columns={name: ds[name] for name in ds.column_names}
        )
        print(f"{os.path.basename(path)}")
        print(f"  file-only verification ok         : {file_only['ok']} (shared={file_only['shared_groups']})")
        print(f"  data-backed verification ok       : {data_backed['ok']} (errors={data_backed['errors']})")

    # Export a readable report summary alongside the manifests.
    summary_path = os.path.join(OUTPUT_DIR, "split_report_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "group_key_rule": GROUP_KEY_RULE,
                "two_way": report.to_dict(),
                "three_way": out3.split_report.to_dict(),
                "temporal": outt.split_report.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(f"\nWrote {manifest_2way}, {manifest_3way} and {summary_path}")
    print("\nAll acceptance checks passed.")


if __name__ == "__main__":
    main()
