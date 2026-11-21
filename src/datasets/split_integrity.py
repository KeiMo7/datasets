# Copyright 2025 The HuggingFace Datasets Authors.
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

# Lint as: python3
"""Split provenance, deterministic re-splitting, overlap detection and cache verification.

This module is opt-in: nothing here runs unless the user passes the corresponding
keyword argument to ``load_dataset``, ``load_from_disk``, ``save_to_disk`` or calls
one of the helpers directly. The default loading path is left untouched.

Concepts
--------
* **Content fingerprint** - an order-sensitive hash of every sample of a split. It is
  computed from the Arrow values themselves, never from file names, timestamps or
  sizes, so replacing a cache file with different data is always detected.
* **Sample identity** - overlap detection works on a *sample identifier* (one or
  several columns chosen by the caller), never on whole-row content equality. Two
  rows with the same content but different identifiers are **not** reported as
  overlapping by default; two rows with the same identifier are, even if their
  content differs. This is configurable via ``SplitIntegrityConfig.id_columns``.
* **Split manifest** - a JSON document listing the dataset identifier, revision,
  split method (seed, ratios, identifier columns), per-split row counts, the content
  fingerprint of each split, and (optionally) the explicit list of sample identifiers
  per split. It is meant to be checked into an experiment directory so that someone
  else can replay the exact split.
"""

import dataclasses
import hashlib
import heapq
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pyarrow as pa

from . import config
from .exceptions import (
    FingerprintMismatchError,
    MissingSplitError,
    SplitManifestError,
    SplitOverlapError,
)
from .utils.logging import get_logger


logger = get_logger(__name__)

SPLIT_INTEGRITY_VERSION = 1
DEFAULT_TRAIN = "train"
DEFAULT_VALIDATION = "validation"
DEFAULT_TEST = "test"
DEFAULT_ID_COLUMN = "id"


# --------------------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------------------


def _fingerprint_sample_id(sample_id: Any) -> str:
    """Stable textual form of a sample identifier, used both for hashing and display."""
    if isinstance(sample_id, (list, tuple)):
        return json.dumps([_fingerprint_sample_id(item) for item in sample_id], ensure_ascii=False)
    if isinstance(sample_id, Mapping):
        return json.dumps({str(k): _fingerprint_sample_id(v) for k, v in sample_id.items()}, sort_keys=True)
    if isinstance(sample_id, bytes):
        return sample_id.hex()
    if isinstance(sample_id, bool):
        return "true" if sample_id else "false"
    if isinstance(sample_id, float) and sample_id.is_integer():
        # `1` and `1.0` identify the same sample; normalise to one token.
        return str(int(sample_id))
    return str(sample_id)


def _row_ids_from_batch(batch: pa.Table, id_columns: Union[str, Sequence[str]]) -> list[Any]:
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    columns = [batch.column(name).to_pylist() for name in id_columns]
    if len(columns) == 1:
        return columns[0]
    return [tuple(values) for values in zip(*columns)]


def _row_digest(values: Any) -> bytes:
    # `pa.Table.to_pylist()` returns an insertion-ordered dict whose key order follows
    # the schema, which is exactly the column order we want to fingerprint.
    encoded = json.dumps(values, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(len(encoded).to_bytes(8, "big") + encoded).digest()


def _batch_fingerprint(batch: pa.Table) -> bytes:
    """Return one digest per row of ``batch``.

    The digest only involves the *values* of the row, never the batch it came from, so
    the split fingerprint is the same whatever the batch size, shard layout or
    compression.
    """
    return [_row_digest(row) for row in batch.to_pylist()]


class ContentFingerprint:
    """Incremental, order-sensitive fingerprint of a dataset split.

    The hash is defined over the column names once and then over the rows in order, so
    it is independent of how rows are grouped into batches. Use
    :func:`compute_content_fingerprint` unless you need to feed batches yourself.
    """

    def __init__(self, column_names: Optional[Sequence[str]] = None):
        self._hasher = hashlib.sha256()
        self._columns = None
        self._num_samples = 0
        if column_names is not None:
            self._set_columns(column_names)

    def _set_columns(self, column_names: Sequence[str]) -> None:
        if self._columns is None:
            self._columns = list(column_names)
            header = json.dumps(self._columns, ensure_ascii=False).encode("utf-8")
            self._hasher.update(b"\x00")
            self._hasher.update(len(header).to_bytes(8, "big") + header)

    def update(self, batch: pa.Table) -> None:
        self._set_columns(batch.column_names)
        self._update_row_digests(_batch_fingerprint(batch))

    def _update_row_digests(self, row_digests: Sequence[bytes]) -> None:
        for digest in row_digests:
            self._hasher.update(digest)
        self._num_samples += len(row_digests)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def iter_arrow_batches(dataset, batch_size: int = 1000) -> Iterable[pa.Table]:
    """Yield raw Arrow tables of ``dataset`` in row order, ignoring the display format.

    The dataset's ``set_format`` / ``with_format`` is intentionally bypassed: the
    fingerprint must describe the stored data, not the formatting currently set by
    the user, and it must not materialise the full table in memory.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    if getattr(dataset, "_indices", None) is not None:
        # Reordered / filtered view: go through __getitem__ in Arrow format so the
        # indices mapping is honoured. Batches are contiguous slices, so memory is
        # bounded by batch_size. `with_format` returns a shallow copy and doesn't
        # mutate the caller's dataset.
        arrow_view = dataset.with_format("arrow")
        for start in range(0, len(arrow_view), batch_size):
            yield arrow_view[start : start + batch_size]
        return
    from .table import table_iter

    yield from table_iter(dataset.data, batch_size=batch_size)


def compute_content_fingerprint(dataset, batch_size: int = 1000, num_proc: Optional[int] = None) -> str:
    """Return the order-sensitive content fingerprint of a split.

    Args:
        dataset ([`Dataset`]): split to fingerprint. It is never modified.
        batch_size (`int`, defaults to `1000`): number of rows hashed at a time. It only
            affects memory use and speed, never the resulting fingerprint.
        num_proc (`int`, *optional*): if greater than 1, batches are hashed by worker
            threads and combined in batch order. The result is identical to the
            sequential one.

    Notes:
        The fingerprint depends only on the column names, the column order and the
        values of each row, in order. It does not depend on the shard layout, the
        number of row groups, file names, modification times or byte sizes.
    """
    if num_proc is not None and num_proc > 1:
        return _compute_content_fingerprint_parallel(dataset, batch_size=batch_size, num_proc=num_proc)

    hasher = ContentFingerprint(column_names=dataset.data.column_names)
    for batch in iter_arrow_batches(dataset, batch_size=batch_size):
        hasher.update(batch)
    return hasher.hexdigest()


def _compute_content_fingerprint_parallel(dataset, batch_size: int, num_proc: int) -> str:
    # Threads, not processes: hashing releases the GIL inside hashlib and pyarrow's
    # to_pylist, and this keeps the result independent of the process start method
    # (spawn/fork) and of the caller's __main__ module. At most `num_proc * 4` batches
    # are in flight; results are consumed in submission (= row) order, so memory stays
    # bounded and the fingerprint is bit-for-bit the sequential one.
    from concurrent.futures import ThreadPoolExecutor

    hasher = ContentFingerprint(column_names=dataset.data.column_names)
    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        window = []
        for batch in iter_arrow_batches(dataset, batch_size=batch_size):
            window.append(executor.submit(_batch_fingerprint, batch))
            if len(window) >= num_proc * 4:
                hasher._update_row_digests(window.pop(0).result())
        while window:
            hasher._update_row_digests(window.pop(0).result())
    return hasher.hexdigest()


def compute_sample_identifiers(
    dataset,
    id_columns: Union[str, Sequence[str]] = DEFAULT_ID_COLUMN,
    batch_size: int = 1000,
) -> list[Any]:
    """Return the ordered list of sample identifiers of a split.

    Raises:
        MissingSplitError: if one of the identifier columns is absent.
    """
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    missing = [name for name in id_columns if name not in dataset.column_names]
    if missing:
        raise MissingSplitError(
            f"The split {dataset.split or '<unnamed>'} is missing the identifier column(s) {missing}. "
            f"Available columns: {dataset.column_names}."
        )
    identifiers: list[Any] = []
    for batch in iter_arrow_batches(dataset, batch_size=batch_size):
        identifiers.extend(_row_ids_from_batch(batch, id_columns))
    return identifiers


# --------------------------------------------------------------------------------------
# Configuration and manifest
# --------------------------------------------------------------------------------------


@dataclass
class SplitIntegrityConfig:
    """Knobs for the opt-in split checks.

    Args:
        enabled (`bool`, defaults to `False`): master switch. When ``False`` every
            caller keeps its historical behaviour.
        id_columns (`str` or `list[str]`, defaults to ``"id"``): columns used as the
            sample identifier for overlap detection. When several columns are given the
            identifier is the tuple of their values.
        require_id_column (`bool`, defaults to ``True``): when the identifier column is
            absent, raise instead of silently skipping overlap detection. Set to
            ``False`` to fingerprint splits whose samples have no explicit identifier
            (overlap detection is then reported as not applicable).
        required_splits (`list[str]`, *optional*): splits that must be present. Missing
            ones raise :class:`MissingSplitError` naming each missing split.
        split_names (`list[str]`, *optional*): splits considered for overlap detection.
            Defaults to ``train`` / ``validation`` / ``test`` when present, else all
            splits.
        batch_size (`int`, defaults to `1000`): rows per chunk for fingerprinting and
            overlap detection. Does not affect any result, only memory/speed.
        num_proc (`int`, *optional*): worker threads for fingerprinting.
        fail_on_overlap (`bool`, defaults to ``True``): raise :class:`SplitOverlapError`
            when overlaps are detected. When ``False`` the report is returned/logged
            instead.
        max_reported_overlaps (`int`, defaults to ``20``): number of sample identifiers
            included per split pair in the report. Full counts are always reported.
        sample_examples (`int`, defaults to ``5``): number of examples included in the
            human-readable error message.
        verify_on_load (`bool`, defaults to ``True``): compare the data read from disk
            with the recorded fingerprint and refuse to load on mismatch.
        strict_missing_manifest (`bool`, defaults to ``True``): when
            ``verify_on_load=True`` and the directory has no manifest, raise instead of
            skipping the check.
        manifest_filename (`str`, defaults to ``"split_integrity.json"``): name of the
            manifest file written next to ``dataset_info.json``.
        manifest_path (`str`, *optional*): where ``load_dataset`` exports the manifest
            of the loaded splits. When ``None`` the manifest is only kept in memory and
            attached to the returned datasets as ``_split_integrity``.
        include_sample_ids (`bool`, defaults to ``True``): include the explicit sample
            identifier list of every split in exported manifests.
        record_cache_manifest (`bool`, defaults to ``True``): write/verify a manifest
            sidecar in the builder cache directory on ``load_dataset``.
    """

    enabled: bool = False
    id_columns: Union[str, Sequence[str]] = DEFAULT_ID_COLUMN
    require_id_column: bool = True
    required_splits: Optional[Sequence[str]] = None
    split_names: Optional[Sequence[str]] = None
    batch_size: int = 1000
    num_proc: Optional[int] = None
    fail_on_overlap: bool = True
    max_reported_overlaps: int = 20
    sample_examples: int = 5
    verify_on_load: bool = True
    strict_missing_manifest: bool = True
    check_overlap_on_load: bool = False
    manifest_filename: str = config.SPLIT_INTEGRITY_FILENAME
    manifest_path: Optional[str] = None
    include_sample_ids: bool = True
    record_cache_manifest: bool = True

    def normalized_id_columns(self) -> list[str]:
        return [self.id_columns] if isinstance(self.id_columns, str) else list(self.id_columns)

    @classmethod
    def coerce(cls, value: Union[bool, "SplitIntegrityConfig", Mapping, None]) -> "SplitIntegrityConfig":
        """Build a config from the value of an opt-in keyword argument."""
        if value is None or value is False:
            return cls(enabled=False)
        if value is True:
            return cls(enabled=True)
        if isinstance(value, cls):
            return value if value.enabled else dataclasses.replace(value, enabled=True)
        if isinstance(value, Mapping):
            unknown = set(value) - {f.name for f in dataclasses.fields(cls)}
            if unknown:
                raise ValueError(
                    f"Unknown split integrity option(s) {sorted(unknown)}. "
                    f"Valid options: {sorted(f.name for f in dataclasses.fields(cls))}."
                )
            return cls(enabled=True, **value)
        raise TypeError(
            "Expected a bool, a SplitIntegrityConfig or a mapping of options, "
            f"got {type(value).__name__}."
        )


@dataclass
class SplitManifest:
    """Reproducible description of how a dataset was split and what it contains."""

    dataset_id: str
    revision: Optional[str] = None
    split_method: str = "as_provided"
    seed: Optional[int] = None
    ratios: Optional[dict[str, float]] = None
    id_columns: list[str] = field(default_factory=lambda: [DEFAULT_ID_COLUMN])
    num_bytes: Optional[dict[str, int]] = None
    num_samples: dict[str, int] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    sample_ids: Optional[dict[str, list[Any]]] = None
    files: Optional[dict[str, list[str]]] = None
    split_integrity_version: int = SPLIT_INTEGRITY_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "SplitManifest":
        if not isinstance(data, Mapping):
            raise SplitManifestError(f"Expected a JSON object for the split manifest, got {type(data).__name__}.")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise SplitManifestError(f"Unsupported split manifest field(s): {sorted(unknown)}.")
        manifest = cls(**data)
        version = manifest.split_integrity_version
        if version != SPLIT_INTEGRITY_VERSION:
            raise SplitManifestError(
                f"Unsupported split manifest version {version!r}, this version of `datasets` understands "
                f"{SPLIT_INTEGRITY_VERSION}."
            )
        return manifest

    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write the manifest as JSON and return the path it was written to."""
        path = os.fspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, config.SPLIT_INTEGRITY_FILENAME)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "SplitManifest":
        path = os.fspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, config.SPLIT_INTEGRITY_FILENAME)
        if not os.path.isfile(path):
            raise SplitManifestError(f"No split manifest found at {path}.")
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def verify(self, dataset_dict, on_missing: str = "error") -> "SplitManifest":
        """Check ``dataset_dict`` against this manifest and refuse on any mismatch.

        Note:
            ``num_bytes`` is deliberately *not* checked: byte sizes depend on the Arrow
            layout, compression and shard count, and are not a property of the sample
            content. Verification uses the order-sensitive content fingerprints and the
            row counts only.

        Args:
            dataset_dict ([`DatasetDict`] or mapping of split name to [`Dataset`]):
                the data to verify.
            on_missing (`str`, defaults to ``"error"``): what to do when a split listed in
                the manifest is absent. ``"error"`` raises :class:`MissingSplitError`.

        Returns:
            The manifest itself, so calls can be chained.
        """
        missing = [name for name in self.num_samples if name not in dataset_dict]
        if missing and on_missing == "error":
            raise MissingSplitError(
                f"Split(s) {missing} are listed in the split manifest but missing from the loaded dataset. "
                f"Available splits: {sorted(dataset_dict)}."
            )
        for name, dataset in dataset_dict.items():
            if name not in self.fingerprints:
                raise FingerprintMismatchError(
                    f"Split '{name}' is not described by the split manifest (manifest splits: "
                    f"{sorted(self.fingerprints)}). Refusing to use a split whose content is unverified."
                )
            expected = self.fingerprints[name]
            actual = compute_content_fingerprint(dataset)
            if actual != expected:
                file_hint = ""
                if self.files and self.files.get(name):
                    file_hint = f" Checked files: {self.files[name]}."
                raise FingerprintMismatchError(
                    f"Content fingerprint mismatch for split '{name}': the manifest records {expected} but the "
                    f"loaded data hashes to {actual}. The cache files were replaced or corrupted.{file_hint}"
                )
            expected_count = self.num_samples.get(name)
            if expected_count is not None and expected_count != len(dataset):
                raise FingerprintMismatchError(
                    f"Sample count mismatch for split '{name}': manifest records {expected_count} samples but "
                    f"{len(dataset)} were loaded."
                )
        return self


# --------------------------------------------------------------------------------------
# Deterministic re-splitting
# --------------------------------------------------------------------------------------


@dataclass
class SplitAssignment:
    """Result of :func:`assign_splits`."""

    assignments: dict[str, list[int]]
    seed: int
    ratios: dict[str, float]
    id_columns: list[str]
    num_samples: int

    def indices_for(self, split: str) -> list[int]:
        return self.assignments[split]

    def counts(self) -> dict[str, int]:
        return {name: len(indices) for name, indices in self.assignments.items()}


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not ratios:
        raise ValueError("ratios must contain at least one split.")
    total = float(sum(ratios.values()))
    if any(value < 0 for value in ratios.values()):
        raise ValueError(f"ratios must be non-negative, got {dict(ratios)}.")
    if total <= 0:
        raise ValueError(f"ratios must sum to a positive number, got {dict(ratios)}.")
    return {str(name): float(value) / total for name, value in ratios.items()}


def assign_splits(
    dataset,
    ratios: Optional[Mapping[str, float]] = None,
    seed: int = 42,
    id_columns: Union[str, Sequence[str]] = DEFAULT_ID_COLUMN,
    split_names: Optional[Sequence[str]] = None,
) -> SplitAssignment:
    """Deterministically assign every row to one split, based only on its sample id.

    The assignment of a sample depends only on ``(seed, split name, sample identifier)``
    and on the (content-adjusted) ratios. It does **not** depend on the number of rows,
    on the row order, on how many worker processes were used, or on how the dataset was
    sharded: appending or removing other samples never moves a given sample to another
    split. Replaying with the same seed and the same ids therefore reproduces the exact
    same split, row for row.

    Each identifier is hashed once with SHA-256; the first 8 bytes give an integer in
    ``[0, 1)`` which is compared with the cumulative ratios in ``split_names`` order.
    Sample identifiers that repeat inside the dataset are assigned together (they are
    the same sample).

    Args:
        dataset ([`Dataset`]): dataset to assign.
        ratios (`Mapping[str, float]`, defaults to ``{"train": 0.8, "validation": 0.1, "test": 0.1}``):
            relative sizes of the splits. Normalised to sum to one.
        seed (`int`, defaults to ``42``): seed of the assignment.
        id_columns (`str` or `list[str]`, defaults to ``"id"``): column(s) forming the
            sample identifier.
        split_names (`Sequence[str]`, *optional*): order in which the ratio buckets are
            consumed. Defaults to ``list(ratios)``.

    Returns:
        [`SplitAssignment`]: contiguous row indices per split, in row order.
    """
    if ratios is None:
        ratios = {DEFAULT_TRAIN: 0.8, DEFAULT_VALIDATION: 0.1, DEFAULT_TEST: 0.1}
    ratios = _validate_ratios(ratios)
    split_names = list(split_names) if split_names is not None else list(ratios)
    if set(split_names) != set(ratios):
        raise ValueError(f"split_names {split_names} don't match the ratio keys {list(ratios)}.")

    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    identifiers = compute_sample_identifiers(dataset, id_columns=id_columns)

    # Precompute the bucket boundaries once. A sample lands in the first bucket whose
    # cumulative upper bound is strictly greater than its uniform value.
    cumulative = []
    running = 0.0
    for name in split_names:
        running += ratios[name]
        cumulative.append((name, running))

    assigned: dict[str, dict[Any, str]] = {}
    assignments: dict[str, list[int]] = {name: [] for name in split_names}
    for row_index, sample_id in enumerate(identifiers):
        split = _bucket_for_sample(sample_id, seed, cumulative, split_names, assigned)
        assignments[split].append(row_index)

    return SplitAssignment(
        assignments=assignments,
        seed=seed,
        ratios={name: ratios[name] for name in split_names},
        id_columns=id_columns,
        num_samples=len(identifiers),
    )


def _uniform_value(sample_id: Any, seed: int) -> float:
    token = f"{seed}\x1f{_fingerprint_sample_id(sample_id)}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _bucket_for_sample(sample_id, seed, cumulative, split_names, cache) -> str:
    key = _fingerprint_sample_id(sample_id)
    if key not in cache:
        value = _uniform_value(sample_id, seed)
        for name, upper in cumulative:
            if value < upper:
                cache[key] = name
                break
        else:
            cache[key] = split_names[-1]
    return cache[key]


def split_by_seed(
    dataset,
    ratios: Optional[Mapping[str, float]] = None,
    seed: int = 42,
    id_columns: Union[str, Sequence[str]] = DEFAULT_ID_COLUMN,
) -> "DatasetDict":
    """Return a [`DatasetDict`] whose splits are the deterministic seed-based assignment.

    This is a convenience wrapper around :func:`assign_splits` plus ``Dataset.select``.
    The original dataset and the default split algorithms are untouched; the returned
    splits carry their provenance in ``_split_integrity``.
    """
    from .dataset_dict import DatasetDict

    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    assignment = assign_splits(dataset, ratios=ratios, seed=seed, id_columns=id_columns)
    split_datasets = {}
    for name, indices in assignment.assignments.items():
        split = dataset.select(indices)
        split._split_integrity = {
            "split": name,
            "split_method": "seeded",
            "seed": seed,
            "ratios": dict(assignment.ratios),
            "id_columns": id_columns,
        }
        split_datasets[name] = split
    return DatasetDict(split_datasets)


# --------------------------------------------------------------------------------------
# Overlap detection
# --------------------------------------------------------------------------------------


def _id_hash(sample_id: Any) -> bytes:
    return hashlib.sha256(_fingerprint_sample_id(sample_id).encode("utf-8")).digest()[:16]


@dataclass
class OverlapReport:
    """Machine-readable result of :func:`detect_split_overlap`.

    Per-split ``num_unique_ids`` and ``duplicate_ids_within_split`` are computed while
    aggregating: the occurrence records are sorted/merged on disk, so the report doesn't
    keep one in-memory entry per sample even for million-row splits.
    """

    split_names: list[str]
    num_samples: dict[str, int]
    num_unique_ids: dict[str, int]
    duplicate_ids_within_split: dict[str, int]
    overlap_pairs: dict[str, dict[str, Any]]
    total_overlapping_unique_ids: int
    id_columns: list[str]
    chunk_size: int
    truncated: bool = False

    @property
    def has_overlap(self) -> bool:
        return self.total_overlapping_unique_ids > 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        if not self.has_overlap:
            return (
                f"No overlap across {len(self.split_names)} split(s) "
                f"({', '.join(f'{name}: {self.num_samples.get(name, 0)} rows' for name in self.split_names)})."
            )
        lines = [
            f"Found {self.total_overlapping_unique_ids} sample id(s) shared across splits "
            f"(id column(s): {self.id_columns})."
        ]
        for pair, info in self.overlap_pairs.items():
            lines.append(
                f"  {pair}: {info['num_overlapping_ids']} shared id(s) "
                f"(rows: {info['num_rows']}); examples: {info['sample_ids']}"
            )
        return "\n".join(lines)


def detect_split_overlap(
    dataset_dict: Mapping[str, Any],
    id_columns: Union[str, Sequence[str]] = DEFAULT_ID_COLUMN,
    chunk_size: int = 1000,
    split_names: Optional[Sequence[str]] = None,
    max_reported_overlaps: int = 20,
) -> OverlapReport:
    """Detect sample identifiers that appear in more than one split.

    Comparison is by sample identifier (``id_columns``), not by whole-row content: two
    rows with the same identifier always count as the same sample even if their other
    columns differ, and two rows with identical content but different identifiers are
    not reported.

    Data is processed in chunks of ``chunk_size`` rows. Chunks are disjoint, exact row
    ranges and every row writes one ``(identifier hash, split, identifier)`` record to a
    temporary spool. The records are then sorted and merged by identifier hash, so all
    occurrences of an identifier - including occurrences on either side of a chunk
    boundary, or spread over many chunks - are aggregated together. No identifier is
    ever dropped or double-counted, memory stays bounded by ``chunk_size`` rather than
    by the dataset size, and any ``chunk_size >= 1`` produces the same report.

    Args:
        dataset_dict: mapping of split name to [`Dataset`].
        id_columns (`str` or `list[str]`, defaults to ``"id"``): sample identifier.
        chunk_size (`int`, defaults to ``1000``): rows read (and buffered) at a time.
        split_names (`Sequence[str]`, *optional*): splits to compare. Defaults to
            ``train``/``validation``/``test`` when present, else all splits.
        max_reported_overlaps (`int`, defaults to ``20``): identifiers kept per split
            pair. Full counts are always reported; the rest is summarised.

    Returns:
        [`OverlapReport`]: per-split row counts, per-pair overlap counts and examples.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    if split_names is None:
        preferred = [name for name in (DEFAULT_TRAIN, DEFAULT_VALIDATION, DEFAULT_TEST) if name in dataset_dict]
        split_names = preferred if preferred else list(dataset_dict)
    split_names = list(split_names)
    missing = [name for name in split_names if name not in dataset_dict]
    if missing:
        raise MissingSplitError(
            f"Split(s) {missing} are required for overlap detection but missing. "
            f"Available splits: {sorted(dataset_dict)}."
        )
    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)

    num_samples: dict[str, int] = {}
    for name in split_names:
        num_samples[name] = len(dataset_dict[name])

    # The per-identifier occurrence table can be large (one entry per distinct sample),
    # so it is spilled to a temporary file and aggregated by a bounded-memory external
    # sort. Memory therefore scales with `chunk_size`, not with the dataset size, and the
    # result is a pure function of the (identifier, split) rows, never of chunking.
    with _OccurrenceSpool(chunk_size=chunk_size) as spool:
        for split_index, name in enumerate(split_names):
            _spool_occurrences(dataset_dict[name], split_index, id_columns, chunk_size, spool)
        return spool.build_report(
            split_names=split_names,
            num_samples=num_samples,
            id_columns=id_columns,
            chunk_size=chunk_size,
            max_reported_overlaps=max_reported_overlaps,
        )


def _spool_occurrences(dataset, split_index, id_columns, chunk_size, spool):
    """Write one fixed-size record per row to the spool, chunk by chunk.

    Records are ``(identifier_hash, split_index, display_id)``. Every row produces one
    record, so an identifier straddling two chunks yields two records that are merged
    later by sorting. No boundary state is kept between chunks and no identifier is
    dropped: chunking only decides how many rows are read at once.
    """
    for batch in iter_arrow_batches(dataset, batch_size=chunk_size):
        for sample_id in _row_ids_from_batch(batch, id_columns):
            spool.write(_id_hash(sample_id), split_index, _display_id(sample_id))


class _OccurrenceSpool:
    """External sort/aggregate of ``(id hash, split)`` records with bounded memory."""

    def __init__(self, chunk_size: int):
        self.chunk_size = max(chunk_size, 1)
        # Read in `chunk_size` rows at a time (that is the user-visible knob), but sort
        # in larger runs so that a million rows don't produce a thousand open streams.
        self.run_size = max(self.chunk_size, 100_000)
        self._file = tempfile.TemporaryFile()
        self._pending: list[tuple[bytes, int, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

    def write(self, key: bytes, split_index: int, display_id: Any) -> None:
        self._pending.append((key, split_index, display_id))
        if len(self._pending) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        for key, split_index, display_id in self._pending:
            record = json.dumps(
                [key.hex(), split_index, display_id], ensure_ascii=False, default=str, separators=(",", ":")
            )
            self._file.write(record.encode("utf-8") + b"\n")
        self._pending = []

    def build_report(self, split_names, num_samples, id_columns, chunk_size, max_reported_overlaps):
        self._flush()
        # Sort the spool in bounded-memory runs, then merge the runs. Only the merged
        # stream - one identifier group at a time - is held in memory.
        run_paths = self._sorted_runs()
        try:
            return self._merge_runs(run_paths, split_names, num_samples, id_columns, chunk_size, max_reported_overlaps)
        finally:
            for path in run_paths:
                os.unlink(path)

    def _sorted_runs(self) -> list[str]:
        self._file.seek(0)
        run_paths = []
        run: list[str] = []
        run_rows = 0
        for line in self._file:
            run.append(line)
            run_rows += 1
            if run_rows >= self.run_size:
                run_paths.append(_write_sorted_run(run))
                run, run_rows = [], 0
        if run:
            run_paths.append(_write_sorted_run(run))
        if not run_paths:
            # Empty input still needs one stream to merge.
            run_paths.append(_write_sorted_run([]))
        return run_paths

    def _merge_runs(self, run_paths, split_names, num_samples, id_columns, chunk_size, max_reported_overlaps):
        pair_counts: dict[str, dict[str, Any]] = {}
        duplicate_within = dict.fromkeys(split_names, 0)
        unique_counts = dict.fromkeys(split_names, 0)
        overlapping_ids = 0
        current_key = None
        current_splits: set[int] = set()
        current_rows = 0
        current_display = None

        def close_group():
            nonlocal overlapping_ids
            if current_key is None:
                return
            present = sorted(current_splits)
            for split_index in present:
                unique_counts[split_names[split_index]] += 1
            if len(present) == 1:
                if current_rows > 1:
                    duplicate_within[split_names[present[0]]] += 1
                return
            overlapping_ids += 1
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    left, right = split_names[present[i]], split_names[present[j]]
                    pair_name = "__".join(sorted((left, right)))
                    pair = pair_counts.setdefault(
                        pair_name,
                        {"splits": sorted((left, right)), "num_overlapping_ids": 0, "num_rows": 0, "sample_ids": []},
                    )
                    pair["num_overlapping_ids"] += 1
                    pair["num_rows"] += current_rows
                    if len(pair["sample_ids"]) < max_reported_overlaps:
                        pair["sample_ids"].append(current_display)

        streams = [open(path, "rb") for path in run_paths]
        try:
            for line in heapq.merge(*streams):
                key, split_index, display_id = _parse_record(line)
                if key != current_key:
                    close_group()
                    current_key, current_splits, current_rows, current_display = key, set(), 0, None
                current_splits.add(split_index)
                current_rows += 1
                if current_display is None:
                    current_display = display_id
            close_group()
        finally:
            for stream in streams:
                stream.close()

        truncated = any(pair["num_overlapping_ids"] > len(pair["sample_ids"]) for pair in pair_counts.values())
        for pair in pair_counts.values():
            # The merge walks identifiers in hash order; sort the retained examples so
            # the report doesn't depend on hash ordering or chunk size.
            pair["sample_ids"] = sorted(pair["sample_ids"], key=_display_sort_key)
        return OverlapReport(
            split_names=split_names,
            num_samples=num_samples,
            num_unique_ids=unique_counts,
            duplicate_ids_within_split=duplicate_within,
            overlap_pairs=dict(sorted(pair_counts.items())),
            total_overlapping_unique_ids=overlapping_ids,
            id_columns=id_columns,
            chunk_size=chunk_size,
            truncated=truncated,
        )


def _parse_record(line: bytes) -> tuple[str, int, Any]:
    key, split_index, display_id = json.loads(line)
    return key, split_index, display_id


def _write_sorted_run(lines: list[bytes]) -> str:
    """Sort ``lines`` by identifier hash and write them to a private temp file.

    Records are JSON arrays whose first field is the hex identifier hash, so sorting (and
    merging) them by their raw bytes groups every occurrence of an identifier together
    and keeps the order deterministic for repeated identifiers.
    """
    fd, path = tempfile.mkstemp(prefix="datasets_split_overlap_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.writelines(sorted(lines))
    except BaseException:
        if os.path.exists(path):
            os.unlink(path)
        raise
    return path


def _display_id(sample_id):
    if isinstance(sample_id, (str, int, float, bool)) or sample_id is None:
        return sample_id
    return _fingerprint_sample_id(sample_id)


def _display_sort_key(sample_id):
    """Deterministic, human-friendly ordering for displayed identifiers."""
    if isinstance(sample_id, bool) or sample_id is None:
        return (0, str(sample_id))
    if isinstance(sample_id, (int, float)):
        return (1, float(sample_id), str(sample_id))
    return (2, _fingerprint_sample_id(sample_id))


def check_split_integrity(
    dataset_dict: Mapping[str, Any],
    config: Optional[Union[bool, SplitIntegrityConfig, Mapping]] = None,
    raise_on_overlap: Optional[bool] = None,
) -> tuple[OverlapReport, dict[str, str]]:
    """Run missing-split, fingerprint and overlap checks in one call.

    Returns:
        ``(report, fingerprints)`` where ``report`` is an [`OverlapReport`] and
        ``fingerprints`` maps each split to its content fingerprint.
    """
    config = SplitIntegrityConfig.coerce(config if config is not None else True)
    missing = [name for name in (config.required_splits or []) if name not in dataset_dict]
    if missing:
        raise MissingSplitError(
            f"Required split(s) {missing} are missing. Available splits: {sorted(dataset_dict)}."
        )
    fingerprints = {
        name: compute_content_fingerprint(dataset, batch_size=config.batch_size, num_proc=config.num_proc)
        for name, dataset in dataset_dict.items()
    }
    id_columns = config.normalized_id_columns()
    split_names = list(config.split_names) if config.split_names is not None else list(dataset_dict)
    available = set().union(*(dataset.column_names for dataset in dataset_dict.values())) if dataset_dict else set()
    missing_id_columns = [name for name in id_columns if name not in available]
    if missing_id_columns and config.require_id_column:
        raise MissingSplitError(
            f"Cannot run overlap detection: identifier column(s) {missing_id_columns} not found. "
            f"Available columns: {sorted(available)}. Pass `id_columns=...` to choose the sample identifier, "
            "or `require_id_column=False` to only fingerprint the splits."
        )
    if missing_id_columns:
        report = OverlapReport(
            split_names=split_names,
            num_samples={name: len(dataset_dict[name]) for name in split_names},
            num_unique_ids={},
            duplicate_ids_within_split={},
            overlap_pairs={},
            total_overlapping_unique_ids=0,
            id_columns=id_columns,
            chunk_size=config.batch_size,
        )
    else:
        report = detect_split_overlap(
            dataset_dict,
            id_columns=id_columns,
            chunk_size=config.batch_size,
            split_names=config.split_names,
            max_reported_overlaps=config.max_reported_overlaps,
        )
    fail = config.fail_on_overlap if raise_on_overlap is None else raise_on_overlap
    if report.has_overlap and fail:
        raise SplitOverlapError(_overlap_error_message(report, config), report=report)
    return report, fingerprints


def _overlap_error_message(report: OverlapReport, config: SplitIntegrityConfig) -> str:
    # Cap the examples in the message itself: for a million-row leak the summary must
    # stay a handful of lines plus a sample, never one line per overlapping sample.
    max_examples = max(config.sample_examples, 1)
    parts = [
        "Split overlap detected: the same sample identifier appears in more than one split. "
        "Training on overlapping splits invalidates evaluation metrics.",
        f"{report.total_overlapping_unique_ids} overlapping sample id(s), {len(report.overlap_pairs)} split pair(s).",
    ]
    for pair_name, info in report.overlap_pairs.items():
        shown = info["sample_ids"][:max_examples]
        parts.append(
            f"  {pair_name}: {info['num_overlapping_ids']} shared id(s), {info['num_rows']} row(s); "
            f"examples: {shown}" + (" ..." if info["num_overlapping_ids"] > len(shown) else "")
        )
    parts.append(
        f"(id column(s): {report.id_columns}, chunk size: {report.chunk_size}). "
        "The full report - still with per-pair counts and capped examples - is on the exception's `.report` attribute."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Manifest construction and disk verification
# --------------------------------------------------------------------------------------


def build_split_manifest(
    dataset_dict: Mapping[str, Any],
    dataset_id: Optional[str] = None,
    revision: Optional[str] = None,
    split_method: str = "as_provided",
    seed: Optional[int] = None,
    ratios: Optional[Mapping[str, float]] = None,
    id_columns: Union[str, Sequence[str]] = DEFAULT_ID_COLUMN,
    include_sample_ids: bool = True,
    batch_size: int = 1000,
    num_proc: Optional[int] = None,
    files: Optional[dict[str, list[str]]] = None,
    extra: Optional[dict] = None,
) -> SplitManifest:
    """Describe a dataset's splits in a reproducible way.

    Args:
        dataset_dict: mapping of split name to [`Dataset`].
        dataset_id (`str`, *optional*): dataset identifier. Defaults to the
            ``dataset_name`` of the first split, then to its split name.
        revision (`str`, *optional*): dataset version or git revision, if known.
        split_method (`str`, defaults to ``"as_provided"``): human-readable description
            of how the split was produced, e.g. ``"seeded"`` or ``"as_provided"``.
        seed (`int`, *optional*): seed used by a seeded split.
        ratios (`mapping`, *optional*): ratios used by a seeded split.
        id_columns (`str` or `list[str]`, defaults to ``"id"``): sample identifier.
        include_sample_ids (`bool`, defaults to ``True``): store the explicit identifier
            list of each split, so someone else can replay and diff it.
        files (`dict`, *optional*): cache file names per split. Recorded for diagnostics,
            never used to compute the fingerprint.
    """
    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    if dataset_id is None:
        for dataset in dataset_dict.values():
            dataset_id = dataset.info.dataset_name
            if dataset_id:
                break
        dataset_id = dataset_id or "unknown"

    manifest = SplitManifest(
        dataset_id=dataset_id,
        revision=None if revision is None else str(revision),
        split_method=split_method,
        seed=seed,
        ratios=None if ratios is None else {str(k): float(v) for k, v in ratios.items()},
        id_columns=id_columns,
        num_samples={},
        fingerprints={},
        sample_ids={} if include_sample_ids else None,
        files={} if files is not None else None,
        extra=dict(extra or {}),
    )
    for name, dataset in dataset_dict.items():
        manifest.num_samples[name] = len(dataset)
        manifest.fingerprints[name] = compute_content_fingerprint(
            dataset, batch_size=batch_size, num_proc=num_proc
        )
        if include_sample_ids:
            if all(column in dataset.column_names for column in id_columns):
                manifest.sample_ids[name] = compute_sample_identifiers(
                    dataset, id_columns=id_columns, batch_size=batch_size
                )
            else:
                # No explicit identifier on this split: the content fingerprint still
                # protects it, but there is no id list to replay. Record that fact
                # instead of failing the whole save.
                manifest.sample_ids[name] = None
                logger.warning(
                    f"Split '{name}' has no identifier column(s) {id_columns} (columns: {dataset.column_names}); "
                    "the manifest records its fingerprint but no per-sample identifiers."
                )
    if files is not None:
        manifest.files = {name: list(files.get(name, [])) for name in dataset_dict}
    return manifest


def cache_files_by_split(dataset) -> list[str]:
    """Return the on-disk cache file names of a split (empty for in-memory datasets)."""
    try:
        return [entry["filename"] for entry in dataset.cache_files]
    except Exception:  # pragma: no cover - defensive, e.g. unusual table backends
        return []


def where_is_sample(dataset_dict: Mapping[str, Any], sample_id: Any, id_columns=DEFAULT_ID_COLUMN) -> dict[str, list[int]]:
    """Return ``{split_name: [row indices]}`` for every split containing ``sample_id``.

    Useful to act on an overlap report: the report says *which* identifiers leaked,
    this says exactly where they are. Comparison uses the same identifier rules as
    :func:`detect_split_overlap` (``1``, ``1.0`` and ``True`` are the same id).
    """
    id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
    target = _id_hash(sample_id)
    locations: dict[str, list[int]] = {}
    for split_name, dataset in dataset_dict.items():
        if not all(column in dataset.column_names for column in id_columns):
            continue
        offset = 0
        for batch in iter_arrow_batches(dataset, batch_size=1000):
            for local_index, row_id in enumerate(_row_ids_from_batch(batch, id_columns)):
                if _id_hash(row_id) == target:
                    locations.setdefault(split_name, []).append(offset + local_index)
            offset += batch.num_rows
    return locations


def verify_splits_against_manifest(
    dataset_dict: Mapping[str, Any],
    manifest: Union[str, os.PathLike, Mapping, SplitManifest],
    fail_on_missing_manifest: bool = True,
) -> SplitManifest:
    """Verify loaded splits against a manifest loaded from disk, a mapping, or an object."""
    if isinstance(manifest, SplitManifest):
        pass
    elif isinstance(manifest, Mapping):
        manifest = SplitManifest.from_dict(manifest)
    else:
        path = os.fspath(manifest)
        if os.path.isdir(path):
            path = os.path.join(path, config.SPLIT_INTEGRITY_FILENAME)
        if not os.path.isfile(path):
            if fail_on_missing_manifest:
                raise SplitManifestError(f"No split manifest found at {path}; refusing to load unverified data.")
            return None
        manifest = SplitManifest.load(path)
    return manifest.verify(dataset_dict)


def find_manifest_path(root: Union[str, os.PathLike], manifest_filename: str = config.SPLIT_INTEGRITY_FILENAME):
    """Return ``root/<manifest_filename>`` if it exists, else ``None``."""
    path = os.path.join(os.fspath(root), manifest_filename)
    return path if os.path.isfile(path) else None


def load_manifest_from_directory(
    root: Union[str, os.PathLike],
    manifest_filename: str = config.SPLIT_INTEGRITY_FILENAME,
    fail_on_missing_manifest: bool = True,
) -> Optional[SplitManifest]:
    """Load ``root/<manifest_filename>`` or raise :class:`SplitManifestError`."""
    path = find_manifest_path(root, manifest_filename=manifest_filename)
    if path is None:
        if fail_on_missing_manifest:
            raise SplitManifestError(
                f"No split manifest found at {os.path.join(os.fspath(root), manifest_filename)}. "
                "Save the dataset with `save_to_disk(..., split_integrity=True)` or pass "
                "`strict_missing_manifest=False` to load without verification."
            )
        return None
    return SplitManifest.load(path)


def attach_manifest_metadata(dataset, manifest: SplitManifest, split_name: str):
    """Attach provenance to a Dataset without changing its data or return structure.

    The metadata lives in ``dataset._split_integrity`` so that pickling, serialization
    and equality of the underlying data are unaffected.
    """
    dataset._split_integrity = {
        "dataset_id": manifest.dataset_id,
        "revision": manifest.revision,
        "split_method": manifest.split_method,
        "seed": manifest.seed,
        "ratios": manifest.ratios,
        "id_columns": manifest.id_columns,
        "split": split_name,
        "num_samples": manifest.num_samples.get(split_name),
        "fingerprint": manifest.fingerprints.get(split_name),
    }
    return dataset

