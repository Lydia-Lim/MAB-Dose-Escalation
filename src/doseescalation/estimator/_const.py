from typing import Sequence
from ._base import EstimatorBase


class ConstEstimator(EstimatorBase):
    def __init__(self, pred_value=0.0):
        self._pred_value = pred_value

    def fit(self, x, y):
        pass

    def predict(self, x) -> Sequence[float]:
        return [self._pred_value] * len(x)
