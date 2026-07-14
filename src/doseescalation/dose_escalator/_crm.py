from typing import Sequence
from doseescalation.estimator import EstimatorBase

from ._base import DoseEscalatorBase
from ._validate import Validator, NoOpValidator, NoSkipValidator


class CRMDoseEscalator(DoseEscalatorBase):
    """
    Continual Reassessment Method based dose escalator.
    This class implicitly models the dose-toxicity curve,
    updates it with observations and makes dose proposals
    according to the model belief.
    Convervative CRM only proposes doses with expected probability
    of dose limiting event no higher than target toxicity level.
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        estimator: EstimatorBase,
        conservative: bool = True,
        no_skip: bool = True,
    ):
        self._dose_levels = dose_levels
        self._ttl = target_toxicity_level
        self._estimator = estimator
        self._conservative = conservative
        self._validator: Validator = (
            NoSkipValidator(len(dose_levels)) if no_skip
            else NoOpValidator()
        )

    def propose(self) -> int:
        # batch predict the probabilities of dose limiting events
        proposed_p_dles = self._estimator.predict(self._dose_levels)

        # iterate over the dose levels to find the dose closest to TTL
        prop_idx = 0
        for idx, p_dle_i in enumerate(proposed_p_dles):
            # check the index is valid according to the validation mechanism
            if not self._validator.validate(idx):
                continue

            # for conservative CRM, only use dose <= TTL
            if self._conservative:
                if (
                    p_dle_i <= self._ttl and (
                        proposed_p_dles[prop_idx] > self._ttl or
                        self._dose_levels[idx]
                        > self._dose_levels[prop_idx]
                    )
                ):
                    prop_idx = idx
            else:
                if (
                    abs(self._ttl - p_dle_i)
                    <= abs(self._ttl - proposed_p_dles[prop_idx])
                ):
                    prop_idx = idx

        return prop_idx

    def safe_doses(self):
        """Per-dose boolean mask of doses whose estimated toxicity is <= TTL."""
        return [p_dle <= self._ttl
                for p_dle in self._estimator.predict(self._dose_levels)]

    def update(self, dose_level_index: int, cohort_size: int, n_dle: int):
        self._validator.visit(dose_level_index)
        self._estimator.fit(
            ((cohort_size, self._dose_levels[dose_level_index]),),
            (n_dle,)
        )
