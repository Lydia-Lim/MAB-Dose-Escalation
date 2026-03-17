from abc import abstractmethod, ABC
from typing import Sequence, Tuple


class EstimatorBase(ABC):
    """
    Base class for the estimators used in this study.
    We deliberately design the interface to resemble sklearn style classes,
    so that they can be integrated more easily.

    Note: we expect the x-y mapping to be monotonously increasing.
    """

    @abstractmethod
    def fit(self, x: Sequence[Tuple[int, float]], y: Sequence[int]):
        """
        Fit the estimator on the given (input, output) pair.
        """

    @abstractmethod
    def predict(self, x: Sequence[float]) -> Sequence[float]:
        """
        Make predictions on the target input.
        """
