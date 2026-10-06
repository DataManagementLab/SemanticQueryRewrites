"""Trimmed vendored copy of cross_db_benchmark.benchmark_tools.generate_workload.

Only the enum definitions needed by the Postgres plan parser are kept; the full
workload generator (and its heavy module-level imports: column_types, utils, numpy
sampling) is dropped.

DELTA vs upstream: ``Operator`` gains ``GT='>'`` and ``LT='<'``. Upstream collapses
``>``→GEQ and ``<``→LEQ, but the ZeroShot feature_statistics encodes ``>`` and ``>=``
(and ``<`` / ``<=``) as *distinct* categorical operators, so the distinction must be
preserved. ``parse_filter`` (vendored alongside) is patched to use GT/LT.
"""

from enum import Enum


class Operator(Enum):
    NEQ = '!='
    EQ = '='
    LEQ = '<='
    LT = '<'
    GEQ = '>='
    GT = '>'
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NOT_NULL = 'IS NOT NULL'
    IS_NULL = 'IS NULL'
    IN = 'IN'
    BETWEEN = 'BETWEEN'

    def __str__(self):
        return self.value


class Aggregator(Enum):
    AVG = 'AVG'
    SUM = 'SUM'
    COUNT = 'COUNT'

    def __str__(self):
        return self.value


class ExtendedAggregator(Enum):
    MIN = 'MIN'
    MAX = 'MAX'

    def __str__(self):
        return self.value


class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'

    def __str__(self):
        return self.value
