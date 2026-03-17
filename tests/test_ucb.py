import numpy as np
import pytest
import random
import tensorflow as tf
from doseescalation.dose_escalator import UCBDoseEscalator
from doseescalation.estimator import (
    ConstEstimator,
    DictEstimator,
    TanhEstimator,
)
from doseescalation.simulated_env import SimulatedEnv


@pytest.mark.unit
def test_instantiation():
    UCBDoseEscalator((0, 1, 2), 0.5, ConstEstimator())


@pytest.mark.unit
def test_propose():
    dose_escalator = UCBDoseEscalator((0, 1, 2), 0.5, ConstEstimator())
    dose_escalator.propose()


@pytest.mark.unit
def test_update():
    dose_escalator = UCBDoseEscalator((0, 1, 2), 0.5, ConstEstimator())
    dose_escalator.update(1, 10, 2)


@pytest.mark.unit
def test_propose_after_update():
    # Given
    dose_escalator = UCBDoseEscalator(
        (0, 1, 2),
        0.5,
        DictEstimator({}),
        ucb_coefficient=0.0,
        no_skip=False,
    )
    dose_escalator.update(0, 10, 3)
    dose_escalator.update(1, 10, 4)
    assert dose_escalator.propose() == 1

    # When
    dose_escalator.update(2, 10, 5)

    # Then
    assert dose_escalator.propose() == 2


@pytest.mark.unit
def test_no_skip_propose_after_update():
    # Given
    dose_escalator = UCBDoseEscalator(
        (0, 1, 2),
        0.5,
        DictEstimator({}),
        ucb_coefficient=0.0,
        no_skip=True,
    )
    assert dose_escalator.propose() == 0
    dose_escalator.update(0, 10, 3)
    dose_escalator.update(1, 10, 4)
    assert dose_escalator.propose() == 1

    # When
    dose_escalator.update(2, 10, 5)

    # Then
    assert dose_escalator.propose() == 2


@pytest.mark.unit
def test_update_when_not_training():
    # Given
    dose_escalator = UCBDoseEscalator((0, 1, 2), 0.5, ConstEstimator())
    dose_escalator.update(1, 10, 4)

    # When
    dose_escalator.train(False)

    # Then
    with pytest.raises(RuntimeError):
        dose_escalator.update(1, 10, 4)

    # When
    dose_escalator.train(True)

    # Then
    dose_escalator.update(1, 10, 4)


@pytest.mark.unit
def test_propose_in_non_training_mode_after_update():
    # Given
    dose_escalator = UCBDoseEscalator(
        (0, 1, 2),
        0.5,
        DictEstimator({}),
        ucb_coefficient=100.0,
        no_skip=False,
    )
    dose_escalator.update(0, 10, 3)
    dose_escalator.update(1, 10, 4)
    assert dose_escalator.propose() == 2

    # When
    dose_escalator.train(False)

    # Then
    assert dose_escalator.propose() == 1


@pytest.mark.integration
def test_with_simulated_env():
    # Given
    MAX_ITERATIONS = 1000
    CONVERGENCE_THRESHOLD = 3
    COHORT_SIZE = 30
    dose_levels = (0.11, 0.25, 0.53, 0.69, 0.99)
    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(123)

    def dose_toxic_curve(dose):
        return (np.tanh(dose) + 1) / 2
    mtl_index = 2
    ttl = dose_toxic_curve(dose_levels[mtl_index])
    dose_escalator = UCBDoseEscalator(
        dose_levels,
        ttl,
        TanhEstimator(),
    )
    dose_response = SimulatedEnv(dose_levels, dose_toxic_curve)

    # When
    converge_len = 0
    total_steps = 0
    while (
        converge_len < CONVERGENCE_THRESHOLD
        and total_steps < MAX_ITERATIONS
    ):
        proposed_level_index = dose_escalator.propose()
        n_dle = dose_response(proposed_level_index, COHORT_SIZE)
        dose_escalator.update(proposed_level_index, COHORT_SIZE, n_dle)

        # make a testing proposal
        dose_escalator.train(False)
        proposed_level_index = dose_escalator.propose()
        dose_escalator.train(True)

        # make sure our model consistently proposes the correct dose level
        if proposed_level_index == mtl_index:
            converge_len += 1
        else:
            converge_len = 0
        total_steps += 1

    # Then
    assert total_steps < MAX_ITERATIONS
