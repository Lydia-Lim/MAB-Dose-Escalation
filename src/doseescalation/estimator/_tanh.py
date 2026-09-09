from typing import Sequence, Tuple

import numpy as np
import scipy.stats as stats
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from ._base import EstimatorBase
from ._bayesian_base import BayesianEstimatorBase, run_chain


class TanhEstimator(BayesianEstimatorBase):
    """
    A hyperboilc tangent function estimator
    of functional form:
    y = ((tanh(x) + 1) / 2)^beta, beta ~ Exp(alpha).
    By fitting this estimator on observations,
    the distribution of beta will be updated.
    """

    def __init__(
        self,
        alpha_: float = 1.0,
        beta_: float = 1.0,
        num_results: int = 500,
        num_burnin_steps: int = 500,
    ):
        super().__init__()
        self._alpha = alpha_
        self._beta = beta_
        self._num_results = num_results
        self._num_burnin_steps = num_burnin_steps

    @staticmethod
    def _transform(bases, beta):
        power_bases = (tf.math.tanh(
            tf.cast(bases, tf.float32)
        ) + 1) / 2
        return tf.math.pow(power_bases, beta)

    def _get_log_lik(self, total_counts, bases, counts):
        def _log_lik(beta):
            rv_beta = tfd.Gamma(
                concentration=self._alpha,
                rate=self._beta
            )
            probs = self._transform(bases, beta)
            rv_obs = tfd.Binomial(total_count=total_counts, probs=probs)
            return (
                rv_beta.log_prob(beta)
                + tf.reduce_sum(rv_obs.log_prob(counts))
            )
        return _log_lik

    def _sample(self):
        if len(self._x) == 0:
            rv_beta = tfd.Gamma(
                concentration=self._alpha,
                rate=self._beta
            )
            samples = rv_beta.sample(self._num_results)
            return samples

        total_counts, bases = tuple(zip(*self._x))

        samples = run_chain(
            self._get_log_lik(total_counts, bases, self._y),
            self._num_results,
            self._num_burnin_steps
        )
        tf.keras.backend.clear_session()
        return samples

    def _inference(self) -> float:
        samples = self._sample()

        # use posterior mean as the predictive beta
        return tf.math.reduce_mean(samples).numpy()


class TanhIntegrativeEstimator(EstimatorBase):
    """
    Hyperbolic Tanh estimator which calculates the
    posterior of `a` using scipy probability functions
    and normalises the posterior using integrate.quad
    methods.

    Once fitted, self._calculate_posterior() will return
    an approximately normalised posterior given the normalisation
    constants and likelihood values learned in self.fit().

    For prediction, the expected posterior value of `a` is
    calculated and used for subsequent predictions.
    """
    def __init__(
        self,
        prior_scale: float = 1.,
        prior_rate: float = 1.,
        integral_upper_lim: float = 20.,
        integral_samples: int = int(1e7)
    ):
        self._prior = lambda x: stats.gamma.pdf(x,
                                                a=prior_scale,
                                                loc=0,
                                                scale=1/prior_rate)
        self._integral_upper_lim = integral_upper_lim
        self._integral_samples = integral_samples
        self._est_a_hat = prior_scale / prior_rate

    @staticmethod
    def _hyperbolic_tanh(
            dose_levels: Sequence[float], a: float
    ) -> Sequence[float]:
        return ((np.tanh(dose_levels) + 1) / 2) ** a

    def _f_lhood(self,
                 a: float,
                 cohort_size: int,
                 n_dle: int,
                 dose: float):
        return stats.binom.pmf(
            n_dle, cohort_size, self._hyperbolic_tanh(dose, a)
        )

    def _calculate_posterior(self,
                             a: np.array,
                             lhoods: Sequence[float],
                             norms: Sequence[float]) -> float:
        y = self._prior(a)
        for lh in lhoods:
            y *= lh(a)
        for norm in norms:
            y *= 1 / norm
        return y

    def fit(self,
            x: Sequence[Tuple[int, float]] = None,
            y: Sequence[int] = None):
        """
        Recursively creates the likelihood functions for each
        dose, as well as the associated normalisation factors.
        Once fitted, _calculate_posterior() generates the approximate
        normalised posterior, and the expected a_hat parameter (taken
        across the posterior) is calculated and used for prediction.

        Note that this function is not an online method, and
        recalculates the posterior from scratch every time this
        function is rerun.
        """
        if not (x or y):
            return

        lhoods = []
        norms = []

        assert len(x) == len(y)
        num_doses = len(y)
        n_patients, dose_levels = zip(*x)

        for i in range(num_doses):
            lhoods.append(
                lambda a: self._f_lhood(a, n_patients[i], y[i], dose_levels[i])
            )

            # Monte-Carlo Integration (assume a won't be greater than ~10)
            a_samples = np.random.uniform(
                0, self._integral_upper_lim, self._integral_samples
            )
            norm = np.sum(self._calculate_posterior(a_samples, lhoods, norms))
            norms.append(norm)

        # Monte-Carlo Integration (assume a won't be greater than ~10)
        a_samples = np.random.uniform(
            0, self._integral_upper_lim, self._integral_samples
        )
        self._est_a_hat = np.sum(
            self._calculate_posterior(a_samples, lhoods, norms) * a_samples
        )

    def predict(self, x: Sequence[float]) -> Sequence[float]:
        return self._hyperbolic_tanh(x, self._est_a_hat)


class CRMEstimator(TanhIntegrativeEstimator):
    """
    Continual Reassessment Method (CRM) estimator for the SEEDA paper's CRM
    baseline. It reuses ``TanhIntegrativeEstimator``'s one-parameter power
    model ``p_k(a) = ((tanh(d_k) + 1) / 2) ** a`` and prior on ``a``, changing
    only the two things that stop the parent class being usable as a live CRM:

    1. **Online.** It accumulates per-dose sufficient statistics (patients,
       DLTs) across successive ``fit`` calls, because ``CRMDoseEscalator``
       feeds it one cohort at a time (the parent treats each ``fit`` as the
       whole dataset and keeps no history).
    2. **Fast.** The posterior-mean ``a_hat`` is computed by grid quadrature
       over ``a`` instead of the parent's 1e7-sample Monte-Carlo integration,
       which makes per-cohort refitting cheap enough for 1000 x 300-cohort
       trials.

    The paper's CRM prior is ``a ~ Exp(mean 0.5) = Gamma(shape 1, rate 2)``,
    the default here. Sharing one parameter across doses means an un-sampled
    dose still gets a model-based toxicity estimate, unlike a per-dose average.
    """

    def __init__(
        self,
        prior_scale: float = 1.0,
        prior_rate: float = 2.0,
        a_max: float = 20.0,
        n_grid: int = 1000,
    ):
        super().__init__(prior_scale=prior_scale, prior_rate=prior_rate)
        self._a_grid = np.linspace(1e-6, a_max, n_grid)
        self._prior_grid = self._prior(self._a_grid)
        self._n: dict = {}   # dose level -> total patients
        self._s: dict = {}   # dose level -> total DLTs

    def fit(self,
            x: Sequence[Tuple[int, float]] = None,
            y: Sequence[int] = None):
        if not (x or y):
            return
        # Accumulate sufficient statistics per dose:
        for (count, dose), n_dle in zip(x, y):
            self._n[dose] = self._n.get(dose, 0) + count
            self._s[dose] = self._s.get(dose, 0) + n_dle
        # Posterior over the a-grid: log prior + sum of binomial log-likelihoods.
        logpost = np.log(self._prior_grid + 1e-300)
        for dose, n in self._n.items():
            s = self._s[dose]
            p = np.clip(self._hyperbolic_tanh(dose, self._a_grid), 1e-12, 1 - 1e-12)
            logpost += s * np.log(p) + (n - s) * np.log1p(-p)
        w = np.exp(logpost - logpost.max())
        # Posterior-mean a (predict() is inherited and reads _est_a_hat):
        self._est_a_hat = float(np.sum(self._a_grid * w) / np.sum(w))
