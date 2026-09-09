from typing import Sequence

from ._base import DoseEscalatorBase


class ThreePlusThreeDoseEscalator(DoseEscalatorBase):
    """
    Three Plus Three approach to dose escalation.
    This class makes proposals for dosage escalation when
    the number of toxicity events among the evaluated patient
    cohorts follow a set of pre-defined rules.
    If the dosage tested doesn't generate any toxicity, the class
    will remain in the same stage and propose to escalate the dose. (stage 0)
    If there is more than one case of relevant toxicity, the
    class will stop suggesting to escalate, and set the MTD. (stage 2)
    If there is one case of toxicity, the class will re-evaluate
    the same dosage level, and depending on the n_dle results
    suggest to escalate or set the MTD. (stage 1)
    """

    def __init__(self, dose_levels: Sequence[float]):
        self._dose_level_index = 0
        self._stage = 0
        self.n_dose_levels = len(dose_levels)

    @property
    def stopped(self) -> bool:
        # Stage 2 means an MTD has been declared; a real 3 + 3 trial ends here
        # and enrols no more patients (see DoseEscalatorBase.stopped).
        return self._stage == 2

    def propose(self) -> int:
        """
        Propose the index of the next dose to trial.
        """
        return self._dose_level_index

    def update(self, dose_level_index: int, _: int, n_dle: int):
        """
        Update the escalator with the environment feedback.
        """
        stage = self._stage
        if stage == 2:
            return
        if n_dle == 0 and stage == 0:
            self._dose_level_index = dose_level_index + 1
        elif n_dle > 1 and stage == 0:
            self._stage = 2
            # When stage 2 is reached, the MTD is defined
            # as the dosage level prior to that
            self._dose_level_index = max(0, dose_level_index - 1)
        elif n_dle == 1 and stage == 0:
            self._dose_level_index = dose_level_index
            self._stage = 1
        elif n_dle == 0 and stage == 1:
            self._dose_level_index = dose_level_index + 1
        elif n_dle >= 1 and stage == 1:
            self._stage = 2
            # When stage 2 is reached, the MTD is defined
            # as the dosage level prior to that
            self._dose_level_index = max(0, dose_level_index - 1)
        self._dose_level_index = min(
            self.n_dose_levels - 1,
            self._dose_level_index
        )
