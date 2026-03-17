import numpy as np
from typing import Callable, Sequence


class SimulatedEnv:
    """
    Simulated environment for dose escalation studies.
    An example usage can be found in `tests/test_simulated_env.py`.
    """
    def __init__(
        self,
        dose_levels: Sequence[float],
        dose_toxic_curve: Callable[[float], float]
    ) -> None:
        self._dose_levels = dose_levels
        self._dose_toxic_curve = dose_toxic_curve

    def __call__(self, dose_level_index: int, cohort_size: int) -> int:
        p_dle = self._dose_toxic_curve(self._dose_levels[dose_level_index])
        n_dle = np.random.binomial(cohort_size, p_dle)
        return n_dle
