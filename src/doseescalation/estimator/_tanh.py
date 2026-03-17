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
