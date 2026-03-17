import tensorflow as tf
from tensorflow_probability import distributions as tfd

from ._bayesian_base import BayesianEstimatorBase, run_chain


class PowerEstimator(BayesianEstimatorBase):
    """
    A power function estimator of functional form:
    y = x^(e^(beta)), beta ~ N(mu, sigma).
    By fitting this estimator on observations,
    the distribution of beta will be updated.
    """

    def __init__(
        self,
        mu: float = 0.,
        sigma: float = 1.,
        num_results: int = 500,
        num_burnin_steps: int = 500,
    ):
        super().__init__()
        self._mu = mu
        self._sigma = sigma
        self._num_results = num_results
        self._num_burnin_steps = num_burnin_steps

    @staticmethod
    def _transform(bases, beta):
        return tf.math.pow(tf.cast(bases, tf.float32), tf.exp(beta))

    def _get_log_lik(self, total_counts, bases, counts):
        def _log_lik(beta):
            rv_beta = tfd.Normal(loc=self._mu, scale=self._sigma)
            probs = self._transform(bases, beta)
            rv_obs = tfd.Binomial(total_count=total_counts, probs=probs)
            return (
                rv_beta.log_prob(beta)
                + tf.reduce_sum(rv_obs.log_prob(counts))
            )
        return _log_lik

    def _sample(self):
        if len(self._x) == 0:
            return [self._mu]

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
