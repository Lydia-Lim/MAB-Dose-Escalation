import math
from doseescalation.estimator import EstimatorBase
from typing import Sequence

from ._base import DoseEscalatorBase
from ._validate import Validator, NoOpValidator, NoSkipValidator


class UCBDoseEscalator(DoseEscalatorBase):
    """
    Dose escalator maximizing the Upper Confidence Bound.
    Note we implement a safe version here, where if the predicted
    p_dle for a dose is beyond the target toxicity level,
    we discount its Q value by the difference.
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        estimator: EstimatorBase,
        ucb_coefficient: float = 1.0,
        is_training: bool = True,
        no_skip: bool = True,
    ):
        self._dose_levels = dose_levels
        self._ttl = target_toxicity_level
        self._estimator = estimator
        self._n_trials_map = [1] * len(dose_levels)
        self._c = ucb_coefficient
        self._is_training = is_training
        self._validator: Validator = (
            NoSkipValidator(len(dose_levels)) if no_skip
            else NoOpValidator()
        )

    def train(self, is_training: bool):
        self._is_training = is_training

    def propose(self) -> int:
        # batch predict the probabilities of dose limiting events
        p_dles = self._estimator.predict(self._dose_levels)

        # use the scaled dose as the Q value if the expected p_dle
        # is below TTL; otherwise, discount the Q value by the diff
        min_dose = min(self._dose_levels)
        max_dose = max(self._dose_levels)
        qs = []
        for dose, p_dle in zip(self._dose_levels, p_dles):
            # scale the dose to the range [0.5, 1]
            q = (dose - min_dose) / (max_dose - min_dose) / 2 + 0.5

            # calculate the discount for the p_dle over TTL
            discount = 1 if p_dle <= self._ttl else (0.5 - (p_dle - self._ttl))
            qs.append(q * discount)
        n = sum(self._n_trials_map)

        # calculate the upper confidence bounds
        c = self._c if self._is_training else 0.0
        ucbs = [
            qs[idx] + c * math.sqrt(
                math.log(n) / self._n_trials_map[idx]
            ) if self._validator.validate(idx) else 0.0
            for idx in range(len(p_dles))
        ]

        # best action/dose is the one with the highest UCB
        return ucbs.index(max(ucbs))

    def safe_doses(self):
        """Per-dose boolean mask of doses whose estimated toxicity is <= TTL."""
        return [p_dle <= self._ttl
                for p_dle in self._estimator.predict(self._dose_levels)]

    def update(
        self,
        dose_level_index: int,
        cohort_size: int,
        n_dle: int
    ) -> None:
        if not self._is_training:
            raise RuntimeError("Cannot update in non-training mode")

        self._validator.visit(dose_level_index)
        self._n_trials_map[dose_level_index] += 1
        self._estimator.fit(
            ((cohort_size, self._dose_levels[dose_level_index]),),
            (n_dle,)
        )
