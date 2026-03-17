import tensorflow as tf
import tensorflow_probability as tfp
from abc import abstractmethod, abstractstaticmethod
from typing import Callable, List, Sequence, Tuple

from ._base import EstimatorBase


@tf.function
def run_chain(target_log_prob_fn, num_results, num_burnin_steps):
    """
    Run the MCMC chain to draw samples from the posterior distribution.

    Note there is currently a limitation that each run of this function
    will introduce a memory leak of ~30 MB, so make sure you reset the
    environment every once in a while.
    """
    # define the HMC
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=0.1
    )
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(num_burnin_steps * 0.8)
    )

    # sample from the chain
    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=1.0,
        kernel=kernel,
        trace_fn=None
    )
    return samples


class BayesianEstimatorBase(EstimatorBase):
    def __init__(self):
        self._x: List[Tuple[int, float]] = []
        self._y: List[int] = []

    def fit(self, x: Sequence[Tuple[int, float]], y: Sequence[int]):
        self._x += list(x)
        self._y += list(y)

    def predict(self, x: Sequence[float]) -> Sequence[float]:
        beta = self._inference()
        probs = self._transform(x, beta)
        return probs

    def expectation(self, x, expr: Callable):
        betas = self._sample()
        probs = self._transform(
            tf.reshape(tf.cast(x, tf.float32), (-1, 1)),
            tf.reshape(betas, (1, -1)),
        )
        return expr(probs)

    @abstractmethod
    def _inference(self):
        """
        Make inference on the model parameters.
        """

    @abstractmethod
    def _sample(self):
        """
        Sample from the posterior distribution of the
        model parameters.
        """

    @abstractstaticmethod
    def _transform(x, beta):
        """
        Transform input and model parameter into the
        binomial parameter.
        """
