import numpy as np

from ._seeda_plateau_twosided_decoupled import (
    SEEDAPlateauTwoSidedDecoupledDoseEscalator,
)


class SEEDAPlateauTwoSidedDecoupledNoLogDoseEscalator(
    SEEDAPlateauTwoSidedDecoupledDoseEscalator
):
    """
    Same as ``SEEDAPlateauTwoSidedDecoupledDoseEscalator`` (UCB's Plateau + two-sided
    L1 + decoupled L1 coefficient) but with a no-log exploration bonus
    ``F = q_hat + sqrt(c*n/N)`` instead of the paper's log bonus
    ``F = q_hat + sqrt(c*log(n)/N)``. This no-log bonus is the one SEEDA (UCB) uses.

    The no-log bonus is much larger for rarely-sampled doses, so it explores the
    low/under-sampled doses more. That sharpens the efficacy-gap estimates the
    plateau test relies on and raises the dose-3 recommendation a few points in the
    benchmark. Note the no-log F is a *deviation* from paper Eq. 4 (which has log),
    so this variant trades a little fidelity for recommendation accuracy.
    """

    def _calc_model_params(self):

        # Reuse a_hat and the admissible set from the parent:
        a_hat, admissible_set, _, _ = super()._calc_model_params()

        # Swap the log exploration bonus for the no-log bonus (as in SEEDA (UCB)):
        F = self._q_hat + np.sqrt(np.sum(self._N) * self._c / self._N)
        
        return a_hat, admissible_set, F, F[admissible_set]
