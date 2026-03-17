from typing import Dict, Sequence, Tuple

from ._base import EstimatorBase


class AveragingEstimator(EstimatorBase):
    """
    Estimate the function mapping between input and output,
    by tracking the average of the output values.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        default_estimate: float = 0.0
    ):
        self._lr = learning_rate
        self._default = default_estimate
        self._estimates: Dict[float, float] = {}
        self._n_estimates: Dict[float, int] = {}

    def fit(self, x: Sequence[Tuple[int, float]], y: Sequence[int]):
        """
        Update the average estimates using
        new_estimate =
            old_estimate + learning_rate * (new_value - old_estimate)
        """
        for (total_count, base), count in zip(x, y):
            y_i = count / total_count

            if base not in self._n_estimates:
                self._n_estimates[base] = 1
                self._estimates[base] = y_i
            else:
                n_est = self._n_estimates[base] + 1
                old_estimate = self._estimates[base]
                self._n_estimates[base] = n_est
                self._estimates[base] = (
                    old_estimate + self._lr * (y_i - old_estimate) / n_est
                )

    def predict(self, x: Sequence[float]) -> Sequence[float]:
        return [self._estimates.get(x_i, self._default) for x_i in x]
