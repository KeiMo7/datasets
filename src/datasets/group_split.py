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

"""Group-aware (a.k.a. *grouped*) dataset splitting.

A regular random split can put different rows of the same real-world entity
(e.g. the same customer, recorded under different sample ids and collection
times) on both sides of a train/test boundary. The validation score then
measures memorization instead of generalization.

This module splits **groups** rather than rows: every row sharing the same
group key is always assigned to one and only one side. It is deliberately
written as a pure, Dataset-independent module so that

* the split planning has no dependency on the in-memory state of any
  :class:`~datasets.Dataset`, and
* an exported manifest can be re-validated later, in another process or on
  another machine, by :func:`verify_group_split_manifest`.

Only the Python standard library and NumPy (already a hard dependency of
``datasets``) are used.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the split side used when a dataset is split into three sides.
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

#: Policies for null (missing) values in the group columns.
NULL_GROUP_POLICIES = ("error", "drop", "singleton")

#: Canonical description of how group keys are derived. This exact string is
#: embedded in every manifest and every report so that the rule cannot be
#: documented differently in two places (hard constraint n°5).
GROUP_KEY_RULE = (
    "The group key of a row is the JSON object (json.dumps with sort_keys=True, "
    "separators=(',', ':'), ensure_ascii=False, default=str) built from the "
    "configured group columns in the order they were given; null group values "
    "are handled separately according to null_group_policy. The group key "
    "digest is sha256(key.encode('utf-8')).hexdigest() of the canonical JSON "
    "string. Digests and traversal orders never participate in random decisions."
)

#: Schema version of the manifest file written by :func:`write_group_split_manifest`.
MANIFEST_VERSION = 1

# Suffix used for the atomic-write temporary file; leftovers are detected on
# the next run (failure/degradation requirement n°3).
_TMP_SUFFIX = ".tmp"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GroupSplitError(ValueError):
    """Raised when a group-aware split cannot satisfy its hard constraints."""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SplitReport:
    """Summary of a group-aware split.

    Every number in this report is computed from the *actual* assignment of
    groups to sides; the manifest and the returned datasets are built from the
    same assignment, so reported counts can never drift from the real products
    (hard constraint n°2).

    Args:
        group_columns: columns used to derive the group key, in given order.
        null_group_policy: policy applied to rows with null group values.
        null_rows: number of rows with at least one null group value.
        side_names: sides of the split, in canonical order
            (``["train", "test"]`` or ``["train", "validation", "test"]``).
        target_ratios: target fraction of (kept) samples per side.
        actual_ratios: fraction of kept samples actually assigned per side.
        ratio_deviations: ``actual_ratios - target_ratios`` per side.
        group_counts: number of distinct groups assigned per side.
        sample_counts: number of rows assigned per side.
        label_distribution: per-side mapping ``label -> count`` (empty when no
            stratification column was requested).
        feasibility: ``"ok"`` when every target is reached within tolerance,
            ``"degraded"`` otherwise.
        warnings: human-readable notes about feasibility/degradation.
        seed: seed used for reproducible random decisions.
        time_column: column used for the temporal constraint, if any.
        time_order_ok: whether the no-future-sample constraint holds per side
            pair; ``None`` when no time column was given.
        group_key_rule: the canonical :data:`GROUP_KEY_RULE` text.
    """

    group_columns: List[str]
    null_group_policy: str
    null_rows: int
    side_names: List[str]
    target_ratios: Dict[str, float]
    actual_ratios: Dict[str, float]
    ratio_deviations: Dict[str, float]
    group_counts: Dict[str, int]
    sample_counts: Dict[str, int]
    label_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    feasibility: str = "ok"
    warnings: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    time_column: Optional[str] = None
    time_order_ok: Optional[bool] = None
    group_key_rule: str = GROUP_KEY_RULE

    @property
    def ok(self) -> bool:
        return self.feasibility == "ok"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitReport":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Group key derivation
# ---------------------------------------------------------------------------


def _canonical_key(values: Sequence[Any]) -> str:
    """Canonical JSON serialization of one row's group values.

    ``sort_keys=True`` makes the representation independent of dict/object
    traversal order; ``default=str`` keeps the function total for values such
    as ``datetime`` or ``decimal.Decimal`` without adding a dependency.
    """
    return json.dumps(list(values), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def group_key_digest(key: str) -> str:
    """SHA-256 hex digest of a canonical group key string."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:  # pragma: no cover - defensive, non-float cannot be NaN
        pass
    if isinstance(value, np.generic):
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, np.ndarray):
        return bool(np.isnan(value).any()) if value.dtype.kind in "fc" else False
    return False


def build_groups(
    columns: Mapping[str, Sequence[Any]],
    group_columns: Sequence[str],
    null_group_policy: str = "error",
) -> Tuple[Dict[str, List[int]], List[int], int]:
    """Group row indices by canonical group key.

    Args:
        columns: mapping ``column name -> 1D sequence of row values`` (the
            sequence must support integer indexing and ``len``).
        group_columns: one or more column names forming the key.
        null_group_policy:

            - ``"error"``: raise :class:`GroupSplitError` if any row has a null
              value in one of the group columns;
            - ``"drop"``: rows with null group values are excluded from every
              side and counted separately;
            - ``"singleton"``: every null row becomes its own unique group so
              that null values are never collapsed into one shared group.

    Returns:
        ``(groups, dropped_rows, null_rows)`` where ``groups`` maps canonical
        group keys to the sorted list of row indices, ``dropped_rows`` lists
        rows excluded under the ``drop`` policy and ``null_rows`` is the count
        of rows with at least one null group value.
    """
    if null_group_policy not in NULL_GROUP_POLICIES:
        raise GroupSplitError(f"null_group_policy must be one of {NULL_GROUP_POLICIES}, got {null_group_policy!r}")
    missing = [c for c in group_columns if c not in columns]
    if missing:
        raise GroupSplitError(f"Group column(s) not found in the dataset: {missing}")
    if not group_columns:
        raise GroupSplitError("At least one group column is required.")

    n_rows = len(columns[group_columns[0]])
    groups: Dict[str, List[int]] = {}
    dropped_rows: List[int] = []
    null_rows = 0

    for row_idx in range(n_rows):
        values = [columns[c][row_idx] for c in group_columns]
        if any(_is_null(v) for v in values):
            null_rows += 1
            if null_group_policy == "error":
                raise GroupSplitError(
                    f"Found {None!r}/NaN values in group column(s) {list(group_columns)} "
                    f"(first offending row index: {row_idx}). Set null_group_policy to "
                    f"'drop' (exclude such rows) or 'singleton' (each null row gets its "
                    "own group) to handle them explicitly."
                )
            if null_group_policy == "drop":
                dropped_rows.append(row_idx)
                continue
            # singleton: make the key unique; mark it so it can never collide
            # with a canonical key built from real values.
            key = f"__null_singleton__:{row_idx}"
        else:
            key = _canonical_key(values)
        groups.setdefault(key, []).append(row_idx)

    for key in groups:
        groups[key].sort()
    return groups, dropped_rows, null_rows


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------


def _normalize_time(values: Sequence[Any], column_name: str) -> np.ndarray:
    """Convert a time column to a 1-D ``datetime64[ns]`` array.

    Supported inputs: ``datetime64`` arrays, Python ``datetime``/``date``,
    ISO-8601 strings and numeric epoch values (seconds, matching
    ``datetime64[s]``). Null values are rejected with a precise error.
    """
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        normalized = arr.astype("datetime64[ns]")
    elif arr.dtype.kind in "iuf":
        normalized = arr.astype("datetime64[s]").astype("datetime64[ns]")
    elif arr.dtype.kind in ("O", "U", "S"):
        converted = []
        for idx, value in enumerate(arr.tolist()):
            if value is None:
                raise GroupSplitError(
                    f"Time column {column_name!r} contains a null value at row {idx}; "
                    "the temporal constraint requires a timestamp for every row."
                )
            if isinstance(value, (_datetime.datetime, _datetime.date)):
                converted.append(np.datetime64(value.isoformat(), "ns"))
            else:
                try:
                    converted.append(np.datetime64(str(value), "ns"))
                except ValueError as error:
                    raise GroupSplitError(
                        f"Value {value!r} in time column {column_name!r} (row {idx}) cannot "
                        "be parsed as an ISO-8601 timestamp."
                    ) from error
        normalized = np.asarray(converted, dtype="datetime64[ns]")
    else:
        raise GroupSplitError(
            f"Time column {column_name!r} has unsupported dtype {arr.dtype}; use timestamps, "
            "datetimes, ISO strings or numeric epoch values."
        )

    try:
        nat_count = int(np.isnat(normalized).sum())
    except TypeError:  # pragma: no cover - defensive
        nat_count = 0
    if nat_count:
        first = int(np.flatnonzero(np.isnat(normalized))[0])
        raise GroupSplitError(
            f"Time column {column_name!r} contains {nat_count} null/NaT timestamp(s) "
            f"(first at row {first}); the temporal constraint requires a timestamp for every row."
        )
    return normalized


# ---------------------------------------------------------------------------
# Size / ratio resolution
# ---------------------------------------------------------------------------


def _resolve_target_counts(
    side_names: Sequence[str],
    sizes: Mapping[str, Optional[float]],
    n_samples: int,
) -> Dict[str, int]:
    """Resolve per-side sizes (floats or ints) into absolute target counts.

    Sizes are given for every side except ``train``; the train target is the
    complement. Mirrors the rounding convention of
    :meth:`Dataset.train_test_split` (``ceil`` for the non-train sides).
    """
    targets: Dict[str, int] = {}
    remaining = n_samples
    for side in side_names:
        if side == TRAIN:
            continue
        size = sizes.get(side)
        if size is None:
            raise GroupSplitError(f"A size is required for side {side!r}.")
        if isinstance(size, int):
            if not 0 < size < n_samples:
                raise GroupSplitError(
                    f"Size {size} for side {side!r} must be positive and smaller than "
                    f"the number of samples {n_samples}."
                )
            targets[side] = size
        elif isinstance(size, float):
            if not 0.0 < size < 1.0:
                raise GroupSplitError(f"Float size for side {side!r} must be in the (0, 1) range, got {size}.")
            targets[side] = math.ceil(size * n_samples)
        else:
            raise GroupSplitError(f"Invalid size {size!r} ({type(size).__name__}) for side {side!r}.")
        remaining -= targets[side]

    if remaining <= 0:
        raise GroupSplitError(
            "The requested non-train sides leave no samples for the train side; "
            f"resolved counts={targets}, total samples={n_samples}."
        )
    targets[TRAIN] = remaining
    return targets


def _target_ratios_from_counts(target_counts: Mapping[str, int], n_samples: int) -> Dict[str, float]:
    return {side: count / n_samples for side, count in target_counts.items()}


# ---------------------------------------------------------------------------
# Assignment engines
# ---------------------------------------------------------------------------


def _group_time_range(rows: Sequence[int], times: np.ndarray) -> Tuple[np.datetime64, np.datetime64]:
    local = times[np.asarray(rows, dtype=np.int64)]
    return local.min(), local.max()


def _time_cuts_feasible(
    ordered_groups: Sequence[Tuple[str, List[int]]],
    times: np.ndarray,
    k: int,
) -> List[Tuple[int, ...]]:
    """Enumerate ordered cuts that keep every later side strictly in the future.

    ``ordered_groups`` is sorted by (min timestamp, key). A cut is feasible
    when, for every boundary, the latest timestamp of the left part is not
    later than the earliest timestamp of the right part: no row of an earlier
    side is newer than any row of a later side.
    """
    n_groups = len(ordered_groups)
    mins = np.empty(n_groups, dtype="datetime64[ns]")
    maxs = np.empty(n_groups, dtype="datetime64[ns]")
    for i, (_, rows) in enumerate(ordered_groups):
        mins[i], maxs[i] = _group_time_range(rows, times)

    # Prefix maxima / suffix minima of group ranges.
    prefix_max = np.maximum.accumulate(maxs)
    suffix_min = np.empty(n_groups, dtype="datetime64[ns]")
    suffix_min[-1] = mins[-1]
    for i in range(n_groups - 2, -1, -1):
        suffix_min[i] = mins[i] if mins[i] < suffix_min[i + 1] else suffix_min[i + 1]

    feasible: List[Tuple[int, ...]] = []
    if k == 2:
        for cut in range(1, n_groups):
            if prefix_max[cut - 1] <= suffix_min[cut]:
                feasible.append((cut,))
    else:  # k == 3
        for cut1 in range(1, n_groups - 1):
            if prefix_max[cut1 - 1] > suffix_min[cut1]:
                continue
            for cut2 in range(cut1 + 1, n_groups):
                if prefix_max[cut2 - 1] <= suffix_min[cut2]:
                    feasible.append((cut1, cut2))
    return feasible


def _score_assignment(
    assignment: Mapping[str, List[str]],
    group_sizes: Mapping[str, int],
    target_counts: Mapping[str, int],
    side_names: Sequence[str],
    group_labels: Optional[Mapping[str, np.ndarray]] = None,
    n_classes: Optional[int] = None,
) -> Tuple[float, float]:
    """Return ``(sample_score, label_score)``; lower is better.

    The sample score is the total absolute deviation from the target sample
    counts. The label score measures deviation from the per-side label
    proportions implied by the global label distribution (stratification is a
    soft objective; never splitting a group is the hard objective).
    """
    counts = {side: sum(group_sizes[g] for g in assignment[side]) for side in side_names}
    sample_score = float(sum(abs(counts[side] - target_counts[side]) for side in side_names))

    label_score = 0.0
    if group_labels is not None and n_classes:
        for side in side_names:
            side_total = max(counts[side], 1)
            side_counts = np.zeros(n_classes, dtype=np.float64)
            for g in assignment[side]:
                side_counts += group_labels[g]
            global_counts = np.zeros(n_classes, dtype=np.float64)
            for g in group_labels:
                global_counts += group_labels[g]
            target_class = global_counts * (side_total / max(sum(global_counts), 1))
            label_score += float(np.abs(side_counts - target_class).sum())
    return sample_score, label_score


def _greedy_lpt(
    group_order: Sequence[str],
    group_sizes: Mapping[str, int],
    target_counts: Mapping[str, int],
    side_names: Sequence[str],
) -> Dict[str, List[str]]:
    """Largest-first greedy packing of whole groups onto sides.

    Each group goes to the side whose resulting sample count is the furthest
    below its target (largest remaining capacity); ties are broken by the
    canonical side order, which is fixed and never derived from hashes or
    container iteration order.
    """
    assignment: Dict[str, List[str]] = {side: [] for side in side_names}
    loads = dict.fromkeys(side_names, 0)
    for g in group_order:
        best_side = min(
            side_names,
            key=lambda side: (loads[side] + group_sizes[g] - target_counts[side], side_names.index(side)),
        )
        assignment[best_side].append(g)
        loads[best_side] += group_sizes[g]
    return assignment


def _local_search(
    assignment: Dict[str, List[str]],
    group_sizes: Mapping[str, int],
    target_counts: Mapping[str, int],
    side_names: Sequence[str],
    group_labels: Optional[Mapping[str, np.ndarray]],
    n_classes: Optional[int],
) -> Dict[str, List[str]]:
    """Deterministic hill climbing: best single-group move, repeated."""
    current = {side: list(groups) for side, groups in assignment.items()}
    current_score = _score_assignment(current, group_sizes, target_counts, side_names, group_labels, n_classes)
    improved = True
    while improved:
        improved = False
        best_move = None
        best_score = current_score
        # Iterate groups in canonical order for reproducibility.
        for g in sorted(group_sizes):
            owner = next(side for side in side_names if g in current[side])
            for dest in side_names:
                if dest == owner:
                    continue
                candidate = {side: list(groups) for side, groups in current.items()}
                candidate[owner].remove(g)
                candidate[dest].append(g)
                score = _score_assignment(candidate, group_sizes, target_counts, side_names, group_labels, n_classes)
                if score < best_score:
                    best_score = score
                    best_move = (g, owner, dest, candidate)
        if best_move is not None:
            _, _, _, candidate = best_move
            current = candidate
            current_score = best_score
            improved = True
    return current


def plan_group_split(
    groups: Mapping[str, List[int]],
    target_counts: Mapping[str, int],
    side_names: Sequence[str],
    seed: int,
    labels_by_row: Optional[Sequence[int]] = None,
    n_classes: Optional[int] = None,
    times: Optional[np.ndarray] = None,
    n_restarts: int = 32,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Assign whole groups to sides.

    Args:
        groups: canonical group key -> row indices (output of
            :func:`build_groups`).
        target_counts: absolute target number of samples per side.
        side_names: canonical side order, length 2 or 3.
        seed: integer seed; required so results are reproducible across
            machines and processes.
        labels_by_row: optional class index per row for the soft
            stratification objective.
        n_classes: number of classes when ``labels_by_row`` is provided.
        times: optional ``datetime64[ns]`` array enabling the temporal
            constraint (earlier sides must never contain newer samples).
        n_restarts: number of randomized greedy restarts (the first,
            restart 0, is always the deterministic largest-first order).

    Returns:
        ``(assignment, warnings)`` where ``assignment`` maps each side to the
        list of canonical group keys it owns.
    """
    if not isinstance(seed, (int, np.integer)):
        raise GroupSplitError("An integer `seed` is required for a reproducible group split.")
    side_names = list(side_names)
    if len(side_names) not in (2, 3):
        raise GroupSplitError("Only 2-side and 3-side group splits are supported.")
    if not groups:
        raise GroupSplitError("No groups available to split (all rows were dropped?).")
    if len(groups) < len(side_names):
        raise GroupSplitError(
            f"Need at least {len(side_names)} groups for {len(side_names)} sides but only "
            f"{len(groups)} group(s) exist; the hard constraint of one side per group cannot be met."
        )

    warnings: List[str] = []
    group_sizes = {g: len(rows) for g, rows in groups.items()}
    total = sum(group_sizes.values())
    if sum(target_counts.values()) != total:
        raise GroupSplitError(
            f"Internal error: target counts sum to {sum(target_counts.values())} but {total} samples are available."
        )

    # Per-group class histograms for the stratification objective.
    group_labels = None
    if labels_by_row is not None and n_classes:
        group_labels = {}
        for g, rows in groups.items():
            hist = np.zeros(n_classes, dtype=np.float64)
            for r in rows:
                hist[int(labels_by_row[r])] += 1
            group_labels[g] = hist

    # ---- Temporal mode: contiguous, time-ordered groups -------------------
    if times is not None:
        ordered = sorted(groups.items(), key=lambda item: (_group_time_range(item[1], times)[0], item[0]))
        feasible_cuts = _time_cuts_feasible(ordered, times, len(side_names))
        if not feasible_cuts:
            raise GroupSplitError(
                "No temporal cut satisfies the no-future-sample constraint: groups overlap in "
                "time such that every ordered boundary would place a newer sample on an earlier side."
            )
        best_assignment = None
        best_score = None
        for cuts in feasible_cuts:
            boundaries = (0,) + cuts + (len(ordered),)
            candidate = {side: [] for side in side_names}
            for j, side in enumerate(side_names):
                candidate[side] = [ordered[i][0] for i in range(boundaries[j], boundaries[j + 1])]
            score = _score_assignment(candidate, group_sizes, target_counts, side_names, group_labels, n_classes)
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = candidate
        return best_assignment, warnings

    # ---- General mode: deterministic LPT + seeded restarts + hill climbing -
    # Canonical, size-descending order. Hash values are never used for
    # randomness; sorting by key only makes the tie-break reproducible.
    canonical_order = sorted(groups, key=lambda g: (-group_sizes[g], g))
    rng = np.random.default_rng(int(seed))

    largest_group = max(group_sizes.values())
    smallest_target = min(target_counts.values())
    if largest_group > smallest_target:
        warnings.append(
            f"Largest group contains {largest_group} samples but the smallest side targets "
            f"{smallest_target} samples; ratio targets are unreachable exactly. The split "
            "returns the closest feasible whole-group assignment and reports the deviation."
        )

    best_assignment = None
    best_score = None
    for restart in range(max(1, n_restarts)):
        if restart == 0:
            order = list(canonical_order)
        else:
            order = list(canonical_order)
            rng.shuffle(order)
        greedy = _greedy_lpt(order, group_sizes, target_counts, side_names)
        refined = _local_search(greedy, group_sizes, target_counts, side_names, group_labels, n_classes)
        score = _score_assignment(refined, group_sizes, target_counts, side_names, group_labels, n_classes)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = refined

    return best_assignment, warnings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def group_split_indices(
    columns: Mapping[str, Sequence[Any]],
    group_columns: Sequence[str],
    side_sizes: Mapping[str, Optional[float]],
    seed: int,
    stratify_by_column: Optional[str] = None,
    time_column: Optional[str] = None,
    null_group_policy: str = "error",
    n_restarts: int = 32,
    ratio_tolerance: float = 0.05,
    strict: bool = False,
) -> Tuple[Dict[str, np.ndarray], SplitReport, Dict[str, List[int]], List[int]]:
    """Compute row indices per side and the :class:`SplitReport`.

    This is the Dataset-independent entry point: it only needs column data,
    so it can be unit-tested and re-run without any ``Dataset`` object.

    Args:
        columns: mapping ``column name -> row values``; must contain every
            column named by ``group_columns``, optionally the stratification
            and time columns.
        group_columns: one or more columns forming the group key.
        side_sizes: mapping of every non-train side to a float ratio or int
            sample count, e.g. ``{"test": 0.2}`` or
            ``{"validation": 0.1, "test": 0.2}``.
        seed: integer seed for reproducibility.
        stratify_by_column: optional :class:`~datasets.ClassLabel` column used
            as a soft stratification objective inside the group constraint.
        time_column: optional column enforcing that earlier sides never
            contain samples newer than later sides.
        null_group_policy: see :func:`build_groups`.
        n_restarts: greedy restarts in :func:`plan_group_split`.
        ratio_tolerance: maximum acceptable absolute ratio deviation per
            side. Exceedance marks the report ``"degraded"`` (or raises in
            ``strict`` mode).
        strict: when ``True``, raise :class:`GroupSplitError` instead of
            returning a degraded report.

    Returns:
        ``(indices, report, groups, dropped_rows)``: per-side sorted row
        index arrays, the report, the group mapping and the dropped row
        indices (empty unless ``null_group_policy="drop"``).
    """
    side_sizes = dict(side_sizes)
    side_names = [TRAIN] + [side for side in (VALIDATION, TEST) if side in side_sizes]
    if VALIDATION in side_sizes and TEST not in side_sizes:
        raise GroupSplitError("A 3-side split requires both 'validation' and 'test' sizes.")

    groups, dropped_rows, null_rows = build_groups(columns, group_columns, null_group_policy)
    n_kept = sum(len(rows) for rows in groups.values())
    if n_kept == 0:
        raise GroupSplitError("No rows remain after applying the null group policy.")

    target_counts = _resolve_target_counts(side_names, side_sizes, n_kept)
    target_ratios = _target_ratios_from_counts(target_counts, n_kept)

    labels_by_row = None
    n_classes: Optional[int] = None
    label_distribution: Dict[str, Dict[str, int]] = {}
    if stratify_by_column is not None:
        if stratify_by_column not in columns:
            raise GroupSplitError(f"Stratification column {stratify_by_column!r} not found in the dataset.")
        label_values = np.asarray(columns[stratify_by_column])
        classes = np.unique(label_values)
        n_classes = len(classes)
        class_index = {int(c): i for i, c in enumerate(classes.tolist())}
        labels_by_row = np.array([class_index[int(v)] for v in label_values], dtype=np.int64)

    times = _normalize_time(columns[time_column], time_column) if time_column is not None else None

    assignment, warnings = plan_group_split(
        groups,
        target_counts,
        side_names,
        seed=seed,
        labels_by_row=labels_by_row,
        n_classes=n_classes,
        times=times,
        n_restarts=n_restarts,
    )

    # Build row indices and all statistics from this single assignment — the
    # same source that feeds the datasets and the manifest.
    indices: Dict[str, np.ndarray] = {}
    group_counts: Dict[str, int] = {}
    sample_counts: Dict[str, int] = {}
    actual_ratios: Dict[str, float] = {}
    ratio_deviations: Dict[str, float] = {}
    for side in side_names:
        side_rows: List[int] = []
        for g in assignment[side]:
            side_rows.extend(groups[g])
        side_rows.sort()
        indices[side] = np.asarray(side_rows, dtype=np.int64)
        group_counts[side] = len(assignment[side])
        sample_counts[side] = len(side_rows)
        actual_ratios[side] = sample_counts[side] / n_kept
        ratio_deviations[side] = actual_ratios[side] - target_ratios[side]
        if stratify_by_column is not None:
            label_distribution[side] = {
                str(int(c)): int((label_values[np.asarray(side_rows, dtype=np.int64)] == c).sum())
                for c in classes.tolist()
            }

    feasibility = "ok"
    if any(abs(d) > ratio_tolerance for d in ratio_deviations.values()):
        feasibility = "degraded"
        msg = (
            "Target ratio unreachable within tolerance "
            f"{ratio_tolerance}: deviations={ {s: round(d, 6) for s, d in ratio_deviations.items()} }."
        )
        warnings.append(msg)
        if strict:
            raise GroupSplitError(msg)

    time_order_ok = None
    if times is not None:
        time_order_ok = True
        for earlier, later in zip(side_names[:-1], side_names[1:]):
            if times[indices[earlier]].max() > times[indices[later]].min():
                time_order_ok = False
                feasibility = "degraded"
                warnings.append(f"Temporal order violated between {earlier!r} and {later!r}; this should not happen.")

    report = SplitReport(
        group_columns=list(group_columns),
        null_group_policy=null_group_policy,
        null_rows=null_rows,
        side_names=list(side_names),
        target_ratios=target_ratios,
        actual_ratios=actual_ratios,
        ratio_deviations=ratio_deviations,
        group_counts=group_counts,
        sample_counts=sample_counts,
        label_distribution=label_distribution,
        feasibility=feasibility,
        warnings=warnings,
        seed=int(seed),
        time_column=time_column,
        time_order_ok=time_order_ok,
    )
    return indices, report, groups, dropped_rows


# ---------------------------------------------------------------------------
# Manifest (export / independent verification / atomicity)
# ---------------------------------------------------------------------------


def _row_manifest_entry(row_idx: int, side: str, group_key: str, dropped: bool = False) -> Dict[str, Any]:
    return {
        "row_index": int(row_idx),
        "side": None if dropped else side,
        "group_key_digest": group_key_digest(group_key),
        "dropped_null_row": bool(dropped),
    }


def build_manifest(
    groups: Mapping[str, List[int]],
    assignment: Mapping[str, List[str]],
    report: SplitReport,
    dropped_rows: Sequence[int] = (),
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the serializable manifest structure from the actual assignment."""
    owner_of_group: Dict[str, str] = {}
    for side, keys in assignment.items():
        for g in keys:
            owner_of_group[g] = side

    rows: List[Dict[str, Any]] = []
    for g in sorted(groups):
        side = owner_of_group[g]
        for row_idx in groups[g]:
            rows.append(_row_manifest_entry(row_idx, side, g))
    for row_idx in sorted(dropped_rows):
        rows.append(
            {
                "row_index": int(row_idx),
                "side": None,
                "group_key_digest": None,
                "dropped_null_row": True,
            }
        )
    rows.sort(key=lambda entry: entry["row_index"])

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "group_key_rule": GROUP_KEY_RULE,
        "report": report.to_dict(),
        "groups": [
            {
                "group_key_digest": group_key_digest(g),
                "side": owner_of_group[g],
                "n_rows": len(groups[g]),
                "row_indices": list(groups[g]),
            }
            for g in sorted(groups)
        ],
        "rows": rows,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_group_split_manifest(manifest: Mapping[str, Any], path: str) -> str:
    """Write the manifest atomically.

    Data first goes to ``path + '.tmp'`` and is then ``os.replace``-d into
    place, so a crashed run never leaves a half-written manifest. A leftover
    ``.tmp`` file from a previous run is detected, reported and removed first.
    """
    tmp_path = path + _TMP_SUFFIX
    if os.path.exists(tmp_path):
        # Residue of an interrupted run: the final manifest was never created.
        os.remove(tmp_path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, fd_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=_TMP_SUFFIX, dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(fd_path, path)
    except BaseException:
        if os.path.exists(fd_path):
            os.remove(fd_path)
        raise
    return path


def verify_group_split_manifest(
    manifest_path: str,
    columns: Optional[Mapping[str, Sequence[Any]]] = None,
    time_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a manifest independently of the split's in-memory state.

    With ``columns=None`` only the manifest itself is checked: every group
    must belong to exactly one side, row indices must be partitioned without
    overlap and the report counts must match the manifest rows.

    With ``columns`` given, group keys are re-derived from the raw data with
    the documented :data:`GROUP_KEY_RULE`, digests are recomputed and the
    temporal constraint is checked when ``time_column`` is provided.

    Returns a dictionary with ``ok`` (bool), ``shared_groups`` (always empty
    on success) and ``errors``/``warnings`` lists. Raises
    :class:`GroupSplitError` only for unreadable/malformed manifest files.
    """
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        raise GroupSplitError(f"Cannot read group split manifest {manifest_path!r}: {error}") from error

    errors: List[str] = []
    warnings: List[str] = []
    result: Dict[str, Any] = {"ok": False, "shared_groups": [], "errors": errors, "warnings": warnings}

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"Manifest version {manifest.get('manifest_version')!r} != supported {MANIFEST_VERSION}.")
    if manifest.get("group_key_rule") != GROUP_KEY_RULE:
        errors.append("The group key rule stored in the manifest differs from the canonical rule.")

    report = SplitReport.from_dict(manifest.get("report", {}))
    groups = manifest.get("groups", [])
    rows = manifest.get("rows", [])

    # 1. One side per group, no cross-side sharing.
    side_of_digest: Dict[str, str] = {}
    shared: List[str] = []
    for entry in groups:
        digest, side = entry["group_key_digest"], entry["side"]
        if digest in side_of_digest and side_of_digest[digest] != side:
            shared.append(digest)
        side_of_digest[digest] = side
    if shared:
        errors.append(f"{len(shared)} group(s) appear on more than one side: {shared[:5]}")
        result["shared_groups"] = shared

    # 2. Row indices form a disjoint partition (dropped rows excluded).
    seen: Dict[int, str] = {}
    overlap_rows: List[int] = []
    for entry in rows:
        if entry.get("dropped_null_row"):
            continue
        idx, side = entry["row_index"], entry["side"]
        if idx in seen:
            overlap_rows.append(idx)
        seen[idx] = side
    if overlap_rows:
        errors.append(f"{len(overlap_rows)} row index(es) assigned to several sides, e.g. {overlap_rows[:5]}.")

    # 3. Report counts must come from the same data as the manifest.
    counted_samples = dict.fromkeys(report.side_names, 0)
    counted_groups = dict.fromkeys(report.side_names, 0)
    for entry in groups:
        if entry["side"] in counted_groups:
            counted_groups[entry["side"]] += 1
            counted_samples[entry["side"]] += int(entry["n_rows"])
    if counted_samples != report.sample_counts:
        errors.append(f"Manifest sample counts {counted_samples} != report {report.sample_counts}.")
    if counted_groups != report.group_counts:
        errors.append(f"Manifest group counts {counted_groups} != report {report.group_counts}.")

    # 4. Digest consistency between the group list and the row list, and
    #    consistency of the per-row side with the owning group's side.
    row_digests = {e["row_index"]: e["group_key_digest"] for e in rows if not e.get("dropped_null_row")}
    group_row_sides: Dict[int, str] = {}
    for entry in groups:
        for r in entry["row_indices"]:
            group_row_sides[r] = entry["side"]
            if row_digests.get(r) != entry["group_key_digest"]:
                errors.append(f"Row {r} digest does not match its group's digest.")
    side_mismatch = [
        e["row_index"]
        for e in rows
        if not e.get("dropped_null_row") and group_row_sides.get(e["row_index"]) != e["side"]
    ]
    if side_mismatch:
        errors.append(
            f"{len(side_mismatch)} row(s) declare a side different from their group's side, "
            f"e.g. rows {side_mismatch[:5]}."
        )
    if sorted(group_row_sides) != sorted(row_digests):
        errors.append("The group list and the row list do not cover the same row indices.")

    # 5. Optional re-derivation from raw data — the strongest check and the
    #    only one that can detect swapped/retroactively changed columns.
    if columns is not None:
        rebuilt, rebuilt_dropped, _ = build_groups(columns, report.group_columns, report.null_group_policy)
        rebuilt_digests = {r: group_key_digest(g) for g, row_ids in rebuilt.items() for r in row_ids}
        mismatch = [r for r, d in rebuilt_digests.items() if row_digests.get(r) != d]
        if mismatch:
            errors.append(
                f"{len(mismatch)} row(s) have group key digests that do not match the re-derived keys, "
                f"e.g. rows {mismatch[:5]}."
            )
        if sorted(rebuilt_dropped) != sorted(e["row_index"] for e in rows if e.get("dropped_null_row")):
            errors.append("Dropped null rows in the manifest do not match the re-derived null rows.")
        if time_column is not None:
            if time_column not in columns:
                errors.append(f"Time column {time_column!r} not available for verification.")
            else:
                times = _normalize_time(columns[time_column], time_column)
                side_rows: Dict[str, List[int]] = {side: [] for side in report.side_names}
                for entry in rows:
                    if not entry.get("dropped_null_row"):
                        side_rows[entry["side"]].append(entry["row_index"])
                for earlier, later in zip(report.side_names[:-1], report.side_names[1:]):
                    if side_rows[earlier] and side_rows[later]:
                        if times[np.asarray(side_rows[earlier])].max() > times[np.asarray(side_rows[later])].min():
                            errors.append(
                                f"Temporal constraint violated between {earlier!r} and {later!r}: a "
                                "training-side sample is newer than a later-side sample."
                            )

    result["ok"] = not errors
    return result
