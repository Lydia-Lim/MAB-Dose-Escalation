from typing import Callable, Sequence
import numpy as np

from ._base import DoseEscalatorBase
from ._validate import Validator, NoOpValidator, NoSkipValidator


class SEEDAOriginalDoseEscalator(DoseEscalatorBase):
    """
    (Original) SEEDA dose escalator — UCB's original version.
    This class uses the SEEDA Method proposed in
    "Learning for Dose Allocation in Adaptive Clinical
    Trials with Safety Constraints". Given a vector of
    assumed toxicity and efficacy probabilities, this class
    proposes doses to assign to subsequent cohorts. Note that
    the proposed dose depends on whether the escalator is
    being trained, and when it is ready to make a suggestion
    towards the end of a trial.
    """

    def __init__(
        self,
        dose_levels: Sequence[float],
        target_toxicity_level: float,
        dose_toxicity_curve: Callable[[Sequence[float], float],
                                      Sequence[float]],
        p_hat: Sequence[float],
        q_hat: Sequence[float],
        ucb_coefficient: float = 1,
        gamma_1: float = 3/2,
        delta_1: float = 0.05,
        is_training: bool = True,
        seed: float = 0,
        no_skip: bool = True,
    ):
        self._dose_levels = dose_levels
        self._ttl = target_toxicity_level
        self._K = len(self._dose_levels)

        self._c = ucb_coefficient
        self._gamma_1 = gamma_1
        self._delta_1 = delta_1
        # Unclear how C_1 is set. Taking the code from the matlab
        # implementation but the logic doesn't align with the paper
        self._C_1 = ((np.min(np.abs(
            -np.log((np.tanh(self._dose_levels) + 1) / 2)
        ))) ** (- 1 / self._gamma_1)) / 30
        self._is_training = is_training

        # Isolated RNG for the per-dose a_hat initialisation:
        self._rng = np.random.RandomState(seed)
        self._a_hat_doses = self._rng.gamma(shape=1, scale=1, size=self._K)
        self._dose_toxicity_curve = dose_toxicity_curve

        assert len(dose_levels) == len(p_hat)
        assert len(dose_levels) == len(q_hat)
        self._p_hat = np.array(p_hat)
        self._q_hat = np.array(q_hat)
        self._I = self._K - 1
        self._N = np.array([1] * self._K)

        self._validator: Validator = (
            NoSkipValidator(len(dose_levels)) if no_skip
            else NoOpValidator()
        )

    def train(self, is_training: bool):
        self._is_training = is_training

    def _calc_model_params(self):
        w = self._N / sum(self._N)
        a_hat = np.dot(w, self._a_hat_doses)

        alpha_t = self._C_1 * self._K * (
            np.log(2 * self._K / self._delta_1) / (2 * sum(self._N))
        ) ** (self._gamma_1 / 2)

        admissible_set = self._dose_toxicity_curve(
            self._dose_levels, a_hat + alpha_t
        ) <= self._ttl
        F = self._q_hat + np.sqrt(sum(self._N) * self._c / self._N)
        F_filtered = F[admissible_set]
        return a_hat, admissible_set, F, F_filtered

    def safe_doses(self):
        """
        Per-dose boolean mask of the doses this design currently declares safe,
        i.e. the admissible set {k : p_k(a_hat + alpha_t) <= theta} (Lemma 1/2).
        Used for the false-alarm / miss-detection error metrics of Figure 1.
        """
        _, admissible_set, _, _ = self._calc_model_params()
        return [bool(x) for x in admissible_set]

    def propose(self) -> int:
        a_hat, admissible_set, F, F_filtered = self._calc_model_params()

        if self._is_training:
            if len(F_filtered) == 0:
                self._I = 0
            else:
                # Pick argmax, or select largest arm if tied:
                for idx, admissible in enumerate(admissible_set):
                    if (admissible and
                            self._validator.validate(idx) and
                            F[idx] >= F[self._I]):
                        self._I = idx
            return self._I
        else:
            p_dle = self._dose_toxicity_curve(self._dose_levels, a_hat)

            # Recommend the highest safe dose by model toxicity (the toxicity
            # MTD); UCB's original SEEDA assumes efficacy is monotonic in dose.
            max_idx = 0
            for idx in range(len(p_dle)):
                if (self._validator.validate(idx) and
                        p_dle[idx] <= self._ttl and
                        p_dle[idx] >= p_dle[max_idx]):
                    max_idx = idx
            return max_idx

    def update(self,
               dose_level_index: int,
               cohort_size: int,
               n_dle: int,
               n_efficate: int):
        """
        Update the escalator with the environment feedback
        """
        if not self._is_training:
            raise RuntimeError("Cannot update in non-training mode")

        self._validator.visit(dose_level_index)
        self._q_hat[dose_level_index] = (
            self._q_hat[dose_level_index] * self._N[dose_level_index]
            + n_efficate
        ) / (self._N[dose_level_index] + cohort_size)
        self._p_hat[dose_level_index] = (
            self._p_hat[dose_level_index] * self._N[dose_level_index]
            + n_dle
        ) / (self._N[dose_level_index] + cohort_size)
        self._N[dose_level_index] = self._N[dose_level_index] + cohort_size
        # Solve dose_toxicity_curve(d, a_hat) = p_hat[i] for a_hat. For curves
        # of the form base^a_hat (base = curve(d, 1)) this inverts in closed
        # form: a_hat = log(p_hat) / log(base). This replaces the Newton solve
        # (from the original code), which for low-toxicity doses could stick at an
        # inflated a_hat (~4-8 vs the true ~1) and make unsafe doses look
        # admissible; the closed form recomputes from the current p_hat each
        # update, so it self-corrects.
        d = self._dose_levels[dose_level_index]
        log_base = np.log(self._dose_toxicity_curve(dose_levels=d, a_hat=1))
        target = float(np.clip(self._p_hat[dose_level_index], 1e-6, 1 - 1e-6))
        self._a_hat_doses[dose_level_index] = np.log(target) / log_base