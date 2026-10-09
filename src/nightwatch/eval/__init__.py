"""评测层：指标、评测集、裁判校准。"""

from .dataset import (
    STRATA, DatasetBuilder, EvalDataset, build_synthetic_dataset,
)
from .metrics import (
    JudgeCalibrator, Metrics, SampleResult, StratifiedReport,
)

__all__ = [
    "Metrics", "SampleResult", "StratifiedReport", "JudgeCalibrator",
    "EvalDataset", "DatasetBuilder", "STRATA", "build_synthetic_dataset",
]
