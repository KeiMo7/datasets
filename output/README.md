# Group-aware split — acceptance artifacts (UID 75)

This directory contains the products and evidence for the group-aware
(grouped) dataset split feature. It addresses the production incident where
the same customer appeared in both train and validation under different
sample ids / collection times, inflating the offline score.

## Files

| File | What it is |
| --- | --- |
| `acceptance_demo.py` | Reproducible end-to-end demo on synthetic leakage-shaped data. Run with `PYTHONPATH=src python3 -P output/acceptance_demo.py` from the repo root. |
| `acceptance_run.log` | Captured output of the demo, with the actual numbers. |
| `manifest_2way.json` | Row-level manifest of the two-way split (each row's side and SHA-256 group key digest, the full report and the canonical group key rule). |
| `manifest_3way.json` | Same for the pairwise group-disjoint train/validation/test split. |
| `split_report_summary.json` | The `SplitReport` objects (target vs actual ratios, deviations, group/sample counts, label distributions) serialized from the exact assignment used to build the datasets. |
| `existing_split_tests.log` | Verbose output of the pre-existing split/stratify/shard tests — all still pass unchanged. |
| `new_group_split_tests.log` | Verbose output of the 32 new tests in `tests/test_group_split.py` — all pass. |

## Where each requirement is covered

### Forms of the split (section 一)

1. One or more group columns, whole groups stay on one side —
   `build_groups` (`src/datasets/group_split.py`) and
   `Dataset.group_train_test_split(..., group_by_columns=[...])`;
   tests `test_groups_collect_all_rows_of_same_key`,
   `test_multiple_group_columns_composite_key`, `test_zero_shared_groups_two_way`.
2. Closest feasible ratio with reported deviation —
   `plan_group_split` (largest-first greedy + seeded restarts + deterministic
   hill climbing) and `SplitReport.ratio_deviations`; test
   `test_ratio_deviation_reported_from_actual_assignment`.
3. Group + stratify together: group integrity is the hard constraint, label
   proportions a soft objective (`_score_assignment`); test
   `test_stratified_grouped_split_keeps_label_distribution_close`.
4. Highly imbalanced groups: feasibility warning + `feasibility="degraded"`
   with measured deviation (never a silently unbalanced result), or raise in
   `strict=True`; test `test_imbalanced_groups_degrade_with_measured_deviation`.
5. Split result, group list and per-side group counts / sample counts / label
   distribution — `SplitReport` and the manifest `groups` list; test
   `test_manifest_covers_every_row_with_digest`.
6. Time column split with no future sample in train —
   `_time_cuts_feasible`; tests `test_temporal_split_has_no_future_sample`,
   `test_temporal_three_way_ordering`.
7. Seeded, cross-machine/cross-process reproducibility — only
   `np.random.default_rng(seed)` drives randomness; test
   `test_same_seed_identical` and
   `test_cross_process_reproducible_across_hash_seeds` (runs child processes
   with three different `PYTHONHASHSEED` values).
8. Train/validation/test, pairwise group-disjoint — test
   `test_three_way_pairwise_disjoint`.
9. State-independent verification — `verify_group_split_manifest` reads only
   the JSON file (and optionally raw columns); manifest tampering tests.
10. Exportable row-level manifest with each row's side and group key digest —
    `write_group_split_manifest`; manifest tests.
11. Existing random/stratified split behavior unchanged —
    `train_test_split` was not modified; see `existing_split_tests.log`
    (27 passed).

### Hard constraints (section 二)

1. Group key hashes and traversal orders never feed random decisions —
   digests are only computed for the manifest; ordering uses explicit
   `sorted(...)` with fixed keys and the RNG uses only the integer seed.
2. Report counts come from the same assignment object used to build the
   datasets and manifest (single source); test
   `test_ratio_deviation_reported_from_actual_assignment`.
3. Dataset object structure and serialization format untouched — the feature
   only calls the existing `Dataset.select`; test
   `test_dataset_structure_and_features_unchanged`.
4. No new dependency — standard library + NumPy only.
5. One canonical group key rule (`GROUP_KEY_RULE`) embedded in every report
   and manifest; test `test_group_key_rule_constant_is_canonical` and
   `test_manifest_roundtrip_and_verification`.

### Failures and degradation (section 三)

1. Null group values: explicit `error`/`drop`/`singleton` policies with a
   separate null count; null rows are never merged into one shared group
   (`test_null_policies`, `test_null_group_policies_on_dataset`).
2. Unreachable target: closest feasible assignment + numerical deviation,
   `feasibility="degraded"` (or raise in strict mode) — never a false success.
3. No half-written manifest on failure: write-temp + `os.replace` atomic
   write, leftover `.tmp` detected and cleaned on rerun —
   `test_atomic_write_no_half_manifest_and_residue_cleanup`.

## Key numbers from the demo run (921 rows, 200 customer groups)

- Two-way split: **0 shared groups**, deviation `{train: 0.0, test: 0.0}`.
- Three-way split: **0 shared groups** for every side pair.
- Temporal split: `max(train time) <= min(test time)` holds; deviation 0.0022.
- Both manifests verify `ok=True` from file alone and against the raw data.
