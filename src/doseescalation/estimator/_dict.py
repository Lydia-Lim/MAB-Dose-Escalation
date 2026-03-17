from typing import Sequence
from ._base import EstimatorBase


class DictEstimator(EstimatorBase):
    def __init__(self, default_map={}):
        self._map = default_map

    def fit(self, x, y):
        for (count, base), y_i in zip(x, y):
            self._map[base] = y_i / count

    def predict(self, x) -> Sequence[float]:
        return [self._map.get(x_i, 1.0) for x_i in x]
