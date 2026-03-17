import numpy as np
import pytest
import random
import tensorflow as tf
from doseescalation.dose_escalator import CRMDoseEscalator
from doseescalation.estimator import (
    ConstEstimator,
    DictEstimator,
    PowerEstimator,
)
from doseescalation.simulated_env import SimulatedEnv


@pytest.mark.unit
def test_instantiation():
    CRMDoseEscalator((0, 1, 2), 0.5, ConstEstimator())


@pytest.mark.unit
def test_propose():
    dose_escalator = CRMDoseEscalator((0, 1, 2), 0.5, ConstEstimator())
    dose_escalator.propose()


@pytest.mark.unit
def test_update():
    dose_escalator = CRMDoseEscalator((0, 1, 2), 0.5, ConstEstimator())
    dose_escalator.update(1, 10, 2)


@pytest.mark.unit
def test_propose_after_update():
    # Given
    dose_escalator = CRMDoseEscalator(
        (0, 1, 2),
        0.5,
        DictEstimator({0: 0.3, 1: 0.4}),
        no_skip=False,
    )
    assert dose_escalator.propose() == 1

    # When
    dose_escalator.update(2, 10, 5)

    # Then
    assert dose_escalator.propose() == 2


@pytest.mark.unit
def test_no_skip_propose_after_update():
    # Given
    dose_escalator = CRMDoseEscalator(
        (0, 1, 2),
        0.5,
        DictEstimator({0: 0.3, 1: 0.4}),
        no_skip=True,
    )
    assert dose_escalator.propose() == 0

    # When
    dose_escalator.update(0, 10, 3)
    dose_escalator.update(1, 10, 4)
    dose_escalator.update(2, 10, 5)

    # Then
    assert dose_escalator.propose() == 2


@pytest.mark.unit
def test_conservative():
    # Given
    dose_escalator = CRMDoseEscalator(
        (10, 20, 30, 40),
        0.5,
        DictEstimator({10: 0.3, 20: 0.4}),
        conservative=True,
        no_skip=False,
    )
    assert dose_escalator.propose() == 1

    # When
    dose_escalator.update(2, 200, 102)
    dose_escalator.update(3, 200, 90)

    # Then
    assert dose_escalator.propose() == 3


@pytest.mark.unit
def test_non_conservative():
    # Given
    dose_escalator = CRMDoseEscalator(
        (10, 20, 30, 40),
        0.5,
        DictEstimator({10: 0.3, 20: 0.4}),
        conservative=False,
        no_skip=False,
    )
    assert dose_escalator.propose() == 1

    # When
    dose_escalator.update(2, 200, 102)
    dose_escalator.update(3, 200, 90)

    # Then
    assert dose_escalator.propose() == 2


@pytest.mark.integration
def test_with_simulated_env():
    # Given
    MAX_ITERATIONS = 1000
    CONVERGENCE_THRESHOLD = 3
    COHORT_SIZE = 30
    dose_levels = (0.13, 0.29, 0.51, 0.71, 0.95)
    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(123)

    def dose_toxic_curve(dose):
        return np.power(dose, 0.5)
    mtl_index = 1
    ttl = dose_toxic_curve(dose_levels[mtl_index])
    dose_escalator = CRMDoseEscalator(
        dose_levels,
        ttl,
        PowerEstimator(),
        conservative=True,
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

        # make sure our model consistently proposes the correct dose level
        if proposed_level_index == mtl_index:
            converge_len += 1
        else:
            converge_len = 0
        total_steps += 1

    # Then
    assert total_steps < MAX_ITERATIONS
