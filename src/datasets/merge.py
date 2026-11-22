# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
"""Schema-aware merging of shards with heterogeneous features.

This module is the single source of truth for the schema unification used by the
three concatenation paths (map-style [`Dataset`], streaming [`IterableDataset`]
and [`DatasetDict`]). It makes every implicit type promotion explicit and
auditable through a [`MergeReport`], and supports declaring that only lossless
promotions are allowed.

The public entry points are:

* [`unify_features`]: compute the unified [`Features`] for a list of feature
  dictionaries and the per-column promotion report, without touching any data.
* [`unify_table`]: cast one pyarrow table to the unified features, including
  [`ClassLabel`] re-indexing and null-filling of missing columns.
* [`MergeStrategy`]: the strategy controlling which promotions are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import pyarrow as pa

from .exceptions import DatasetsError
from .features.features import ClassLabel, Features, LargeList, List, Value, _is_null_feature
from .table import _short_str, cast_table_to_features


if TYPE_CHECKING:
    from .features.features import FeatureType


# Numeric promotion ladders, ordered from narrow to wide.
_SIGNED_INT_ORDER = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
_UNSIGNED_INT_ORDER = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
_FLOAT_ORDER = [pa.float16(), pa.float32(), pa.float64()]
_STRING_FAMILY = (pa.string(), pa.large_string(), pa.string_view())
_BINARY_FAMILY = (pa.binary(), pa.large_binary(), pa.binary_view())


class MergeError(DatasetsError):
    """Raised when shards cannot be merged with the requested [`MergeStrategy`].

    Args:
        column_name (`str`):
            Name of the column that can't be unified.
        feature_types (`list`):
            The conflicting feature types found across the shards, one per shard
            that has the column (null-only shards are excluded).
        reason (`str`):
            Human-readable explanation of why the unification is impossible or
            forbidden by the strategy.
    """

    def __init__(self, column_name: str, feature_types: list, reason: str):
        self.column_name = column_name
        self.feature_types = feature_types
        self.reason = reason
        rendered_types = ", ".join(_short_str(feature_type) for feature_type in feature_types)
        super().__init__(
            f"Can't merge column '{column_name}': {reason}. Conflicting types across shards: [{rendered_types}]."
        )


class MergeStrategy:
    """Configuration controlling which type promotions are allowed when merging shards.

    Args:
        allow_lossy (`bool`, defaults to `False`):
            If `True`, lossy promotions (e.g. int to float, float32 to float64 is
            lossless, but int64 to int32 is lossy) are allowed and reported as
            lossy in the [`MergeReport`]. If `False` (default, "only lossless
            promotions"), the first lossy or impossible column raises a
            [`MergeError`].
        fill_missing_columns (`bool`, defaults to `True`):
            If `True`, columns missing from a shard are added as nullable columns
            filled with null values (never zeros and never dropping rows). If
            `False`, a column present in some shards but missing in others raises
            a [`MergeError`].

    Example:

    ```python
    >>> from datasets.merge import MergeStrategy
    >>> lossless_only = MergeStrategy()
    >>> allow_lossy = MergeStrategy(allow_lossy=True)
    ```
    """

    def __init__(self, allow_lossy: bool = False, fill_missing_columns: bool = True):
        self.allow_lossy = allow_lossy
        self.fill_missing_columns = fill_missing_columns

    def __eq__(self, other):
        return (
            isinstance(other, MergeStrategy)
            and self.allow_lossy == other.allow_lossy
            and self.fill_missing_columns == other.fill_missing_columns
        )

    def __repr__(self):
        return f"MergeStrategy(allow_lossy={self.allow_lossy}, fill_missing_columns={self.fill_missing_columns})"

    def __hash__(self):
        return hash((self.allow_lossy, self.fill_missing_columns))


# The default strategy only allows lossless promotions and null-fills missing columns.
DEFAULT_MERGE_STRATEGY = MergeStrategy()


@dataclass
class ColumnMergeReport:
    """Report of the unification of a single column.

    Args:
        column_name (`str`):
            Name of the column.
        source_features (list of `FeatureType`):
            Feature of the column in each shard, in shard order. Shards missing
            the column have a `None` entry.
        unified_feature (`FeatureType`):
            The feature all shards are promoted to.
        kind (`str`):
            One of `"identical"` (no promotion needed), `"promotion"` (a type
            promotion is applied), `"class_label_merge"` (class dictionaries are
            merged and indices re-mapped), `"missing_column"` (some shards don't
            have the column, null-filled) and `"null_promotion"` (a shard only
            contained null values for the column).
        lossy (`bool`):
            Whether the promotion can lose information.
        conversions (list of `(int, str, str, bool)` tuples):
            One entry per shard that needs a conversion:
            `(shard_index, source_type_repr, target_type_repr, lossy)`.
        present_in_shards (list of `int`):
            Indices of the shards that contain the column.
    """

    column_name: str
    source_features: list[Optional["FeatureType"]]
    unified_feature: "FeatureType"
    kind: str
    lossy: bool
    conversions: list[tuple[int, str, str, bool]] = field(default_factory=list)
    present_in_shards: list[int] = field(default_factory=list)

    @property
    def is_identical(self) -> bool:
        return self.kind == "identical" and not self.conversions

    def __str__(self):
        if self.is_identical:
            return f"- '{self.column_name}': {_short_str(self.unified_feature)} (identical in all shards)"
        lines = [
            f"- '{self.column_name}': unified to {_short_str(self.unified_feature)} "
            f"[{self.kind}{', LOSSY' if self.lossy else ''}]"
        ]
        for shard_index, source_type, target_type, lossy in self.conversions:
            lines.append(f"    shard {shard_index}: {source_type} -> {target_type}{' (LOSSY)' if lossy else ''}")
        missing = [i for i, feature in enumerate(self.source_features) if feature is None]
        if missing:
            lines.append(f"    shards {missing}: column missing, filled with null")
        return "\n".join(lines)


@dataclass
class MergeReport:
    """Report of the unification of a whole set of shards.

    It answers the two questions "can these shards be merged?" and "is any
    information lost by the merge?".

    Args:
        column_reports (list of [`ColumnMergeReport`]):
            One report per column of the unified features, in column order.
        strategy ([`MergeStrategy`]):
            The strategy that was used to compute the unification.
    """

    column_reports: list[ColumnMergeReport] = field(default_factory=list)
    strategy: MergeStrategy = field(default_factory=MergeStrategy)

    @property
    def lossy(self) -> bool:
        """`True` if at least one column requires a lossy promotion."""
        return any(column_report.lossy for column_report in self.column_reports)

    @property
    def has_promotions(self) -> bool:
        """`True` if at least one column is promoted, null-filled or re-indexed."""
        return any(not column_report.is_identical for column_report in self.column_reports)

    def column_report(self, column_name: str) -> ColumnMergeReport:
        for column_report in self.column_reports:
            if column_report.column_name == column_name:
                return column_report
        raise KeyError(column_name)

    def can_merge(self) -> bool:
        """Whether the shards are compatible under the strategy.

        A report only exists when unification succeeded, so this is always `True`
        for a returned report; it exists to make call sites read naturally.
        """
        return True

    def summary(self) -> str:
        """Render a human-readable summary of the merge."""
        header = (
            "Merge is LOSSY (some columns lose information)"
            if self.lossy
            else "Merge is lossless (no information lost)"
        )
        body = "\n".join(str(column_report) for column_report in self.column_reports)
        return f"{header}\nStrategy: {self.strategy!r}\n{body}"

    def to_dict(self) -> dict:
        """Serializable representation (e.g. for logging or fingerprinting)."""
        return {
            "strategy": {
                "allow_lossy": self.strategy.allow_lossy,
                "fill_missing_columns": self.strategy.fill_missing_columns,
            },
            "lossy": self.lossy,
            "columns": [
                {
                    "column_name": report.column_name,
                    "kind": report.kind,
                    "lossy": report.lossy,
                    "unified_feature": repr(report.unified_feature),
                    "conversions": [
                        {"shard": shard, "from": source, "to": target, "lossy": lossy}
                        for shard, source, target, lossy in report.conversions
                    ],
                    "present_in_shards": report.present_in_shards,
                }
                for report in self.column_reports
            ],
        }


def _index_of(pa_type: pa.DataType, ladder: list[pa.DataType]) -> int:
    for index, candidate in enumerate(ladder):
        if pa_type == candidate:
            return index
    return -1


def _unify_value_types(column_name: str, values: list[Value], strategy: MergeStrategy) -> tuple[Value, bool]:
    """Unify a set of non-null [`Value`] features.

    Returns the unified value and whether the promotion is lossy for at least one
    of the source types.
    """
    pa_types = [Value(value.dtype).pa_type for value in values]

    # Null types are handled by the caller, but keep a defensive check.
    non_null = [pa_type for pa_type in pa_types if not pa.types.is_null(pa_type)]
    if not non_null:
        return values[0], False

    # All the same (non-null) type: no promotion.
    if all(pa_type == non_null[0] for pa_type in non_null):
        first_non_null = next(value for value in values if not _is_null_feature(value))
        return Value(first_non_null.dtype), False

    def fail(reason: str):
        raise MergeError(column_name, list(values), reason)

    # Booleans only promote losslessly among themselves; bool -> int is lossy.
    if any(pa.types.is_boolean(pa_type) for pa_type in non_null):
        if all(pa.types.is_boolean(pa_type) for pa_type in non_null):
            return Value("bool"), False
        if not strategy.allow_lossy:
            fail("boolean can't be losslessly promoted to a numeric type")
        # bool -> the numeric unification of the remaining numeric types.
        numeric = [pa_type for pa_type in non_null if not pa.types.is_boolean(pa_type)]
        if not numeric:
            return Value("int64"), True
        target, numeric_lossy = _unify_value_types(
            column_name, [Value(_dtype_of(pa_type)) for pa_type in numeric], strategy
        )
        # bool -> numeric is itself a lossy promotion, on top of any numeric loss.
        return target, True or numeric_lossy

    signed = [pa_type for pa_type in non_null if pa.types.is_signed_integer(pa_type)]
    unsigned = [pa_type for pa_type in non_null if pa.types.is_unsigned_integer(pa_type)]
    floats = [pa_type for pa_type in non_null if pa.types.is_floating(pa_type)]
    strings = [
        pa_type
        for pa_type in non_null
        if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type) or pa.types.is_string_view(pa_type)
    ]
    binaries = [
        pa_type
        for pa_type in non_null
        if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type) or pa.types.is_binary_view(pa_type)
    ]
    temporals = [pa_type for pa_type in non_null if pa.types.is_temporal(pa_type)]
    decimals = [pa_type for pa_type in non_null if pa.types.is_decimal(pa_type)]

    numeric = signed + unsigned + floats
    # Only one of these type families may be present at a time.
    families = [numeric, strings, binaries, temporals, decimals]
    if sum(1 for family in families if family) > 1:
        fail("incompatible type families (e.g. numeric vs string vs binary vs temporal)")

    if signed and not unsigned and not floats:
        widest = max(signed, key=lambda pa_type: _index_of(pa_type, _SIGNED_INT_ORDER))
        # Widening to the common widest type is lossless; only a source already
        # wider than the target would be lossy, which can't happen here.
        return Value(_dtype_of(widest)), False

    if unsigned and not signed and not floats:
        widest = max(unsigned, key=lambda pa_type: _index_of(pa_type, _UNSIGNED_INT_ORDER))
        return Value(_dtype_of(widest)), False

    if (signed or unsigned) and not floats:
        # Mixed signed/unsigned integers: promote to a signed type wide enough to
        # hold the unsigned range. uint64 has no wider signed 64-bit type.
        max_unsigned_width = max((_index_of(pa_type, _UNSIGNED_INT_ORDER) for pa_type in unsigned), default=-1)
        max_signed_width = max((_index_of(pa_type, _SIGNED_INT_ORDER) for pa_type in signed), default=-1)
        # An unsigned type of width n needs a signed type of width n+1 to be lossless.
        required_signed_width = max_unsigned_width + 1
        target_width = max(max_signed_width, required_signed_width)
        if target_width >= len(_SIGNED_INT_ORDER):
            if not strategy.allow_lossy:
                fail("unsigned 64-bit integers can't be losslessly represented by a signed integer")
            target = pa.int64()
            lossy = True
        else:
            target = _SIGNED_INT_ORDER[target_width]
            lossy = False
        return Value(_dtype_of(target)), lossy

    if floats:
        widest_float = max(floats, key=lambda pa_type: _index_of(pa_type, _FLOAT_ORDER))
        lossy = False
        if signed or unsigned:
            # An integer mixed with a float always lands on a float target.
            # float32 can't hold all integers: require float64, and even there
            # 64-bit integers may lose precision beyond 2^53.
            if widest_float != pa.float64():
                if not strategy.allow_lossy:
                    fail("integers can't be losslessly represented by a narrow float")
                lossy = True
            if any(pa_type == pa.int64() or pa_type == pa.uint64() for pa_type in signed + unsigned):
                if not strategy.allow_lossy:
                    fail("64-bit integers can lose precision when represented as float64")
                lossy = True
            # Any other integer -> float64 is an exact representation (lossless).
        return Value(_dtype_of(widest_float)), lossy

    if strings:
        widest = pa.large_string()
        # string/large_string/string_view hold the same logical values: lossless.
        return Value(_dtype_of(widest)), False

    if binaries:
        widest = pa.large_binary()
        return Value(_dtype_of(widest)), False

    if temporals:
        if all(pa_type == temporals[0] for pa_type in temporals):
            return Value(_dtype_of(temporals[0])), False
        fail("temporal types with different units or timezones can't be unified losslessly")

    if decimals:
        # A decimal is losslessly promoted by widening precision while keeping or
        # increasing the scale; differing scales can't be reconciled losslessly.
        max_precision = max(pa_type.precision for pa_type in decimals)
        scales = {pa_type.scale for pa_type in decimals}
        if len(scales) != 1:
            if not strategy.allow_lossy:
                fail("decimals with different scales can't be unified without rescaling")
            target = pa.decimal256(max_precision + max(scales), max(scales))
            return Value(_dtype_of(target)), True
        target = pa.decimal256(max_precision, scales.pop())
        return Value(_dtype_of(target)), False

    fail("no common type")


def _dtype_of(pa_type: pa.DataType) -> str:
    """Render a pyarrow type the way [`Value`] dtypes are spelled."""
    if pa.types.is_string_view(pa_type):
        return "string_view"
    if pa.types.is_large_string(pa_type):
        return "large_string"
    if pa.types.is_binary_view(pa_type):
        return "binary_view"
    if pa.types.is_large_binary(pa_type):
        return "large_binary"
    if pa.types.is_decimal(pa_type):
        return f"decimal{256 if pa.types.is_decimal256(pa_type) else 128}({pa_type.precision}, {pa_type.scale})"
    return str(pa_type)


def _unify_class_labels(
    column_name: str, labels: list[ClassLabel], strategy: MergeStrategy
) -> tuple[ClassLabel, bool]:
    """Merge several [`ClassLabel`] dictionaries.

    Label texts are preserved row by row: the merged name list is the union of
    the source names (in first-seen order), and each shard's integer indices are
    re-mapped to the merged dictionary. Always lossless.
    """
    merged_names: list[str] = []
    seen: set[str] = set()
    for label in labels:
        for name in label.names:
            if name not in seen:
                seen.add(name)
                merged_names.append(name)
    return ClassLabel(names=merged_names), False


def _unify_two_features(
    column_name: str,
    feature_a: "FeatureType",
    feature_b: "FeatureType",
    strategy: MergeStrategy,
) -> tuple["FeatureType", bool]:
    """Unify two features, returning the unified feature and the lossy flag."""
    if _is_null_feature(feature_a):
        return feature_b, False
    if _is_null_feature(feature_b):
        return feature_a, False

    if isinstance(feature_a, ClassLabel) and isinstance(feature_b, ClassLabel):
        return _unify_class_labels(column_name, [feature_a, feature_b], strategy)

    if isinstance(feature_a, ClassLabel) or isinstance(feature_b, ClassLabel):
        # ClassLabel vs plain string: keep the class dictionary, strings are
        # encoded through ClassLabel.cast_storage. Lossless.
        label, other = (feature_a, feature_b) if isinstance(feature_a, ClassLabel) else (feature_b, feature_a)
        if isinstance(other, Value) and (other.dtype in ("string", "large_string", "string_view")):
            return _unify_class_labels(column_name, [label, ClassLabel(names=label.names)], strategy)
        raise MergeError(
            column_name,
            [feature_a, feature_b],
            "a class dictionary can only be merged with strings or another class dictionary",
        )

    if isinstance(feature_a, Value) and isinstance(feature_b, Value):
        return _unify_value_types(column_name, [feature_a, feature_b], strategy)

    # Containers: List / LargeList must share the same container shape. A mix of
    # List and LargeList is promoted to LargeList (wider offsets, lossless).
    if isinstance(feature_a, (List, LargeList)) and isinstance(feature_b, (List, LargeList)):
        target_container = (
            LargeList if (isinstance(feature_a, LargeList) or isinstance(feature_b, LargeList)) else List
        )
        if (
            isinstance(feature_a, List)
            and feature_a.length != -1
            and isinstance(feature_b, List)
            and feature_b.length != -1
        ):
            if feature_a.length != feature_b.length:
                raise MergeError(
                    column_name,
                    [feature_a, feature_b],
                    "fixed-size lists with different lengths can't be unified",
                )
            length = feature_a.length
        else:
            length = -1
        inner, lossy = _unify_two_features(column_name, feature_a.feature, feature_b.feature, strategy)
        if target_container is LargeList:
            return LargeList(inner), lossy
        if length != -1:
            return List(inner, length=length), lossy
        return List(inner), lossy

    if isinstance(feature_a, dict) and isinstance(feature_b, dict):
        merged: dict[str, FeatureType] = {}
        lossy = False
        for key in feature_a.keys() | feature_b.keys():
            if key in feature_a and key in feature_b:
                sub_feature, sub_lossy = _unify_two_features(
                    f"{column_name}.{key}", feature_a[key], feature_b[key], strategy
                )
                merged[key] = sub_feature
                lossy = lossy or sub_lossy
            elif key in feature_a:
                merged[key] = feature_a[key]
            else:
                merged[key] = feature_b[key]
        return merged, lossy

    raise MergeError(
        column_name,
        [feature_a, feature_b],
        "structurally incompatible features (different shapes or type families)",
    )


def _feature_lossy_relation(
    source: "FeatureType", target: "FeatureType", strategy: MergeStrategy, column_name: str
) -> bool:
    """Whether casting `source` to `target` is lossy.

    Both features are assumed to be unifiable (they come from a successful
    unification); only the lossiness of the concrete promotion is reported.
    """
    if _is_null_feature(source) or source == target:
        return False
    try:
        _, lossy = _unify_two_features(column_name, source, target, MergeStrategy(allow_lossy=True))
    except MergeError:
        lossy = True
    return lossy


def unify_features(
    features_list: list[Features],
    strategy: Optional[MergeStrategy] = None,
) -> tuple[Features, MergeReport]:
    """Compute the unified features for a list of shard features.

    This is a schema-only operation: no row data is read. The same function is
    used by the map-style, streaming and dataset-dictionary concatenation paths
    so their semantics stay identical.

    Args:
        features_list (list of [`Features`]):
            One feature dictionary per shard.
        strategy ([`MergeStrategy`], *optional*):
            Promotion strategy. Defaults to a lossless-only strategy that
            null-fills missing columns.

    Returns:
        tuple of ([`Features`], [`MergeReport`]):
            The unified features (column order follows the first shard, then
            columns only seen in later shards), and a report describing every
            per-column promotion.

    Raises:
        [`MergeError`]: if a column can't be unified, or if a required promotion
            is forbidden by the strategy.
    """
    if strategy is None:
        strategy = DEFAULT_MERGE_STRATEGY
    if not features_list:
        raise ValueError("unify_features requires at least one shard's features")

    # Ordered union of column names: first shard order, then newly seen columns.
    column_order: list[str] = []
    for features in features_list:
        for column_name in features:
            if column_name not in column_order:
                column_order.append(column_name)

    unified = Features()
    reports: list[ColumnMergeReport] = []

    for column_name in column_order:
        source_features: list[Optional[FeatureType]] = [features.get(column_name) for features in features_list]
        present = [index for index, feature in enumerate(source_features) if feature is not None]
        missing = [index for index, feature in enumerate(source_features) if feature is None]

        if missing and not strategy.fill_missing_columns:
            raise MergeError(
                column_name,
                [source_features[index] for index in present],
                f"column is missing in shard(s) {missing} and fill_missing_columns is disabled",
            )

        present_features = [source_features[index] for index in present]
        non_null_features = [
            (index, feature) for index, feature in zip(present, present_features) if not _is_null_feature(feature)
        ]

        if not non_null_features:
            # Every shard that has the column only has null values: keep the
            # null feature and report it.
            target = present_features[0] if present_features else Value("null")
            kind = "null_promotion"
            lossy = False
        else:
            target = non_null_features[0][1]
            lossy = False
            for _, feature in non_null_features[1:]:
                target, pair_lossy = _unify_two_features(column_name, target, feature, strategy)
                lossy = lossy or pair_lossy
            kind = "identical"

        # Build the per-shard conversion list and refine the column kind.
        conversions: list[tuple[int, str, str, bool]] = []
        class_label_merged = isinstance(target, ClassLabel) and any(
            isinstance(source_features[index], ClassLabel) and source_features[index] != target for index in present
        )
        any_promotion = False
        for index in present:
            source = source_features[index]
            if _is_null_feature(source):
                conversions.append((index, repr(source), repr(target), False))
                any_promotion = True
                continue
            if source != target:
                shard_lossy = _feature_lossy_relation(source, target, strategy, column_name)
                if shard_lossy and not strategy.allow_lossy:
                    # Defensive: unify should already have raised above.
                    raise MergeError(
                        column_name,
                        [source, target],
                        "lossy promotion forbidden by the merge strategy",
                    )
                conversions.append((index, repr(source), repr(target), shard_lossy))
                lossy = lossy or shard_lossy
                any_promotion = True

        if class_label_merged:
            kind = "class_label_merge"
        elif missing:
            kind = "missing_column"
        elif any_promotion:
            kind = "promotion"
        elif non_null_features and kind != "null_promotion":
            kind = "identical"

        unified[column_name] = target
        reports.append(
            ColumnMergeReport(
                column_name=column_name,
                source_features=source_features,
                unified_feature=target,
                kind=kind,
                lossy=lossy,
                conversions=conversions,
                present_in_shards=present,
            )
        )

    any_lossy = any(column_report.lossy for column_report in reports)
    if any_lossy and not strategy.allow_lossy:
        lossy_report = next(report for report in reports if report.lossy)
        raise MergeError(
            lossy_report.column_name,
            [feature for feature in lossy_report.source_features if feature is not None],
            "lossy promotion forbidden by the merge strategy (allow_lossy=False)",
        )

    return unified, MergeReport(column_reports=reports, strategy=strategy)


def _reindex_class_label_array(array: pa.ChunkedArray, source: ClassLabel, target: ClassLabel) -> pa.ChunkedArray:
    """Re-map integer indices of a source ClassLabel to the merged dictionary.

    Label texts are preserved: each source index is translated to its label name
    and then to the index of that name in the merged dictionary.
    """
    # Build the old-index -> new-index map; -1 (missing label) stays -1.
    index_map = {index: target._str2int[name] for index, name in enumerate(source.names)}
    index_map[-1] = -1

    def reindex_chunk(chunk: pa.Array) -> pa.Array:
        values = chunk.to_pylist()
        remapped = [None if value is None else index_map[int(value)] for value in values]
        return pa.array(remapped, type=target.pa_type)

    return pa.chunked_array([reindex_chunk(chunk) for chunk in array.chunks])


def unify_table(
    table: pa.Table,
    shard_index: int,
    unified_features: Features,
    report: MergeReport,
) -> pa.Table:
    """Cast one shard table to the unified features.

    Missing columns are added as nullable null arrays (never zeros, rows are
    never dropped); [`ClassLabel`] columns are re-indexed so that label texts
    stay identical row by row; other promotions go through the regular,
    safe-casting machinery (which raises instead of silently truncating).

    Args:
        table (`pyarrow.Table`):
            The shard table to cast.
        shard_index (`int`):
            Position of the shard, used to look up the relevant column reports.
        unified_features ([`Features`]):
            Target features returned by [`unify_features`].
        report ([`MergeReport`]):
            Report returned by [`unify_features`].

    Returns:
        `pyarrow.Table`: the casted table with the unified arrow schema.
    """
    target_schema = unified_features.arrow_schema
    columns = {}
    for column_name in unified_features:
        if column_name not in table.column_names:
            columns[column_name] = pa.chunked_array([pa.nulls(table.num_rows, type=pa.null())])
            continue
        array = table.column(column_name)
        column_report = report.column_report(column_name)
        source_feature = column_report.source_features[shard_index]
        target_feature = unified_features[column_name]
        if (
            isinstance(target_feature, ClassLabel)
            and isinstance(source_feature, ClassLabel)
            and source_feature != target_feature
        ):
            array = _reindex_class_label_array(array, source_feature, target_feature)
            # Rebuild a table with the re-indexed storage before the generic cast.
            table = table.set_column(table.column_names.index(column_name), column_name, array)
        columns[column_name] = table.column(column_name)

    ordered_arrays = [columns[column_name] for column_name in unified_features]
    casted = pa.Table.from_arrays(ordered_arrays, names=list(unified_features))
    if casted.schema != target_schema:
        casted = cast_table_to_features(casted, unified_features)
    else:
        casted = casted.replace_schema_metadata(target_schema.metadata)
    return casted


def merge_strategy_fingerprint(strategy: MergeStrategy) -> str:
    """Stable string identifying a strategy, used to make cache keys strategy-aware."""
    return f"merge_strategy(allow_lossy={strategy.allow_lossy},fill_missing_columns={strategy.fill_missing_columns})"


def unify_example(
    example: dict,
    shard_index: int,
    unified_features: Features,
    report: MergeReport,
) -> dict:
    """Cast a single example (row) of a shard to the unified features.

    This is the row-level (non-arrow) counterpart of [`unify_table`], used by
    streaming sources that don't expose arrow batches. Missing columns are
    null-filled and [`ClassLabel`] integer values are re-indexed through the
    merged dictionary so label texts are preserved.

    Args:
        example (`dict`):
            One row of the shard.
        shard_index (`int`):
            Position of the shard the row comes from.
        unified_features ([`Features`]):
            Target features returned by [`unify_features`].
        report ([`MergeReport`]):
            Report returned by [`unify_features`].

    Returns:
        `dict`: the example with unified, ordered keys.
    """
    output = {}
    for column_name, target_feature in unified_features.items():
        if column_name not in example or example[column_name] is None:
            output[column_name] = None
            continue
        value = example[column_name]
        source_feature = report.column_report(column_name).source_features[shard_index]
        if (
            isinstance(target_feature, ClassLabel)
            and isinstance(source_feature, ClassLabel)
            and source_feature != target_feature
            and isinstance(value, int)
        ):
            # Re-index an integer label; -1 (missing label) is preserved.
            value = -1 if value == -1 else target_feature._str2int[source_feature._int2str[value]]
        output[column_name] = value
    return output
