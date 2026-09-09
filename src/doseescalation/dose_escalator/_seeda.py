from typing import Callable, Sequence
import numpy as np

from ._base import DoseEscalatorBase
from ._validate import Validator, NoOpValidator, NoSkipValidator


class SEEDADoseEscalator(DoseEscalatorBase):
    """
    (Base) SEEDA dose escalator.
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
        ucb_coefficient: float = 1,
        gamma_1: float = 3/2,
        delta_1: float = 0.05,
        is_training: bool = True,
        seed: float = 0,
        no_skip: bool = True,
        a_init=None,
        a_max=None,
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

        self._rng = np.random.RandomState(seed)
        # Per-dose a-estimate. The authors' MATLAB initializes ak_hat = a0
        # (deterministic) and clamps it at a_max (Safe_UCB.m: a0=a_max=20;
        # Safe_UCB_Plateau.m: a0=1/2, a_max=1). When a_init / a_max are None we
        # keep the prior random-gamma init and apply no clamp, so other callers
        # are unaffected.
        if a_init is None:
            self._a_hat_doses = self._rng.gamma(shape=1, scale=1, size=self._K)
        else:
            self._a_hat_doses = np.full(self._K, float(a_init))
        self._a_max = a_max
        self._dose_toxicity_curve = dose_toxicity_curve

        # MATLAB (Safe_UCB.m / Safe_UCB_Plateau.m) uses NO prior: initialize
        # q_hat = p_hat = 0, N = 0, then sample each dose once, setting the
        # estimates from the observed first cohort (see propose / update below).
        self._p_hat = np.zeros(self._K)
        self._q_hat = np.zeros(self._K)
        self._N = np.zeros(self._K, dtype=int)
        self._init_idx = 0  # next dose to sample in the initial round-robin
        self._I = self._K - 1

        self._validator: Validator = (
            NoSkipValidator(len(dose_levels)) if no_skip
            else NoOpValidator()
        )

    def train(self, is_training: bool):
        self._is_training = is_training

    def _calc_model_params(self):
        # MATLAB (Safe_UCB.m / Safe_UCB_Plateau.m) uses a_hat = ak_hat of the
        # MOST-SAMPLED dose ([~,I_est]=max(n_choose); a_hat=ak_hat(I_est)), NOT the
        # paper's weighted average sum(w_k * a_k). Matches both algorithms' code.
        a_hat = self._a_hat_doses[int(np.argmax(self._N))]

        alpha_t = self._C_1 * self._K * (
            np.log(2 * self._K / self._delta_1) / (2 * sum(self._N))
        ) ** (1.0 / (self._gamma_1 * 2.0))

        admissible_set = self._dose_toxicity_curve(
            self._dose_levels, a_hat + alpha_t
        ) <= self._ttl
        # Paper eq. (4): F(p, s, n) = p + sqrt(c * log(n) / s), where n = sum(N),
        # s = N_k. (The no-log bonus variant is SEEDA (UCB).) An unsampled dose
        # (N_k = 0) has an infinite exploration bonus by definition; errstate keeps
        # that intended inf without emitting a divide-by-zero warning. The MATLAB
        # never hits this because it only evaluates f over the admissible set,
        # which it leaves empty until the warm-up round-robin has finished.
        with np.errstate(divide="ignore"):
            F = self._q_hat + np.sqrt(np.log(sum(self._N)) * self._c / self._N)
        F_filtered = F[admissible_set]
        return a_hat, admissible_set, F, F_filtered

    def safe_doses(self):
        """
        Per-dose boolean mask of the doses this design currently declares safe,
        i.e. the admissible set {k : p_k(a_hat + alpha_t) <= theta}. This is the
        SEEDA safety classification used in the false-alarm (Lemma 1) and
        miss-detection (Lemma 2) error metrics of Figure 1.
        """
        # During the initial round-robin sum(N) = 0, so _calc_model_params would
        # divide by zero; no dose is classified until every dose has been sampled.
        if self._init_idx < self._K:
            return [False] * self._K
        _, admissible_set, _, _ = self._calc_model_params()
        return [bool(x) for x in admissible_set]

    def propose(self) -> int:
        # Warm-up (MATLAB Safe_UCB.m): the ALLOCATION samples each dose once in
        # ascending order before any model-based selection. The RECOMMENDATION,
        # however, is always the model's best MTD estimate -- the paper's Algorithm 1
        # output d_hat = argmax_{p_k(a_hat)<=theta} q_hat, which the MATLAB computes
        # every round including warm-up -- so we do NOT short-circuit it to the
        # round-robin dose (that would leak the just-sampled dose into the interim
        # recommendation). Before the first cohort there is no data (sum(N) = 0, so
        # _calc_model_params would divide by zero) -> fall back to the lowest dose.
        if self._is_training and self._init_idx < self._K:
            return self._init_idx
        if not self._is_training and sum(self._N) == 0:
            return 0

        a_hat, admissible_set, F, F_filtered = self._calc_model_params()

        if self._is_training:
            if len(F_filtered) == 0:
                self._I = 0
            else:
                best_idx = None
                for idx, admissible in enumerate(admissible_set):
                    if admissible and self._validator.validate(idx):
                        if best_idx is None or F[idx] >= F[best_idx]:
                            best_idx = idx
                if best_idx is not None:
                    self._I = best_idx
            return self._I
        else:
            # Recommendation (MATLAB Safe_UCB.m: argmax q_hat .* (p_out <= thre)):
            # among doses whose MODEL toxicity p_k(a_hat) <= theta, pick the
            # highest-efficacy one. NB the safety filter uses the model toxicity
            # p_k(a_hat), NOT the raw observed p_hat -- the raw estimate from a few
            # patients on a rarely-sampled high dose is noisy and would let unsafe
            # doses through once the (previously prior-seeded) p_hat is data-only.
            p_out = self._dose_toxicity_curve(self._dose_levels, a_hat)
            max_idx = None
            for idx in range(len(self._q_hat)):
                if (self._validator.validate(idx) and
                        p_out[idx] <= self._ttl):
                    if max_idx is None or self._q_hat[idx] >= self._q_hat[max_idx]:
                        max_idx = idx
            return max_idx if max_idx is not None else 0

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
        # form: a_hat = log(p_hat) / log(base).
        #
        # This replaces the warm-started Newton solve.
        # For low-toxicity doses base^a is nearly flat, so once an early
        # p_hat ~= 0 pushed a_hat large, Newton failed to converge and the
        # except-clause kept the stale (inflated) value forever: a_hat reached
        # ~4-8 vs the true ~1, which made unsafe doses look admissible. The
        # closed form recomputes from the current p_hat each update, so it
        # self-corrects as p_hat converges.
        d = self._dose_levels[dose_level_index]
        log_base = np.log(self._dose_toxicity_curve(dose_levels=d, a_hat=1))

        # Clip to avoid log(0) / underflow issues:
        target = float(np.clip(self._p_hat[dose_level_index], 1e-6, 1 - 1e-6))
        self._a_hat_doses[dose_level_index] = np.log(target) / log_base
        # MATLAB clamps the per-dose a-estimate at a_max (Safe_UCB.m: 20,
        # Safe_UCB_Plateau.m: 1); no clamp when a_max is None.
        if self._a_max is not None:
            self._a_hat_doses[dose_level_index] = min(
                self._a_hat_doses[dose_level_index], self._a_max
            )

        # Advance the initial round-robin once this dose's first cohort has been
        # observed, so the next cohort samples the next dose.
        if self._init_idx < self._K:
            self._init_idx += 1