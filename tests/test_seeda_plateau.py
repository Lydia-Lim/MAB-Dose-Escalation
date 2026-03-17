import numpy as np
import pytest
from doseescalation.dose_escalator import SEEDAPlateauDoseEscalator
from doseescalation.simulated_env import SimulatedEnv


def hyperbolic_tanh(dose_levels, a_hat):
    return ((np.tanh(dose_levels) + 1) / 2) ** a_hat


@pytest.mark.unit
def test_instantiation():
    SEEDAPlateauDoseEscalator((0, 1, 2),
                              0.5,
                              hyperbolic_tanh,
                              (0.2, 0.4, 0.6),
                              (0.2, 0.4, 0.6))


@pytest.mark.unit
def test_propose():
    dose_escalator = SEEDAPlateauDoseEscalator((0, 1, 2),
                                               0.5,
                                               hyperbolic_tanh,
                                               (0.2, 0.4, 0.6),
                                               (0.2, 0.4, 0.6))
    dose_escalator.propose()


@pytest.mark.unit
def test_update():
    dose_escalator = SEEDAPlateauDoseEscalator((0, 1, 2),
                                               0.5,
                                               hyperbolic_tanh,
                                               (0.2, 0.4, 0.6),
                                               (0.2, 0.4, 0.6))
    dose_escalator.update(1, 10, 2, 1)


@pytest.mark.unit
def test_propose_after_update():
    # Given
    dose_escalator = SEEDAPlateauDoseEscalator(
        dose_levels=(0, 0.2, 0.5, 0.9, 1.2),
        target_toxicity_level=0.5,
        dose_toxicity_curve=hyperbolic_tanh,
        p_hat=(0.50, 0.60, 0.73, 0.86, 0.92),
        q_hat=np.array([0.3] * 5),
        eta=50,
        seed=0,
        no_skip=False,
    )
    assert dose_escalator.propose() == 1
    dose_escalator.update(1, 5, 0, 0)
    assert dose_escalator.propose() == 2

    dose_escalator.train(False)
    assert dose_escalator.propose() == 1


@pytest.mark.unit
def test_no_skip_propose_after_update():
    # Given
    dose_escalator = SEEDAPlateauDoseEscalator(
        dose_levels=(0, 0.2, 0.5, 0.9, 1.2),
        target_toxicity_level=0.5,
        dose_toxicity_curve=hyperbolic_tanh,
        p_hat=(0.50, 0.60, 0.73, 0.86, 0.92),
        q_hat=np.array([0.3] * 5),
        eta=50,
        seed=0,
        no_skip=True,
    )
    assert dose_escalator.propose() == 0
    dose_escalator.update(0, 5, 0, 0)
    dose_escalator.update(1, 5, 0, 0)
    assert dose_escalator.propose() == 2
    dose_escalator.update(2, 5, 0, 0)
    assert dose_escalator.propose() == 3

    dose_escalator.train(False)
    assert dose_escalator.propose() == 0


@pytest.mark.integration
def test_with_simulated_env():
    # Given
    MAX_ITERATIONS = 1000
    CONVERGENCE_THRESHOLD = 3
    COHORT_SIZE = 30
    dose_levels = (0.11, 0.25, 0.53, 0.69, 0.99)

    def dose_toxic_curve(dose, a_hat=1):
        return ((np.tanh(dose) + 1) / 2) ** a_hat
    mtl_index = 2
    ttl = dose_toxic_curve(dose_levels[mtl_index]) + 0.05
    dose_escalator = SEEDAPlateauDoseEscalator(
        dose_levels=dose_levels,
        target_toxicity_level=ttl,
        dose_toxicity_curve=hyperbolic_tanh,
        p_hat=(0.50, 0.60, 0.73, 0.86, 0.92),
        q_hat=np.array([0.33] * 5),
        seed=0,
        eta=50
    )
    dose_response = SimulatedEnv(dose_levels, dose_toxic_curve)

    converge_len = 0
    total_steps = 0
    while (
        converge_len < CONVERGENCE_THRESHOLD
        and total_steps < MAX_ITERATIONS
    ):
        proposed_level_index = dose_escalator.propose()
        n_dle = dose_response(proposed_level_index, COHORT_SIZE)
        dose_escalator.update(proposed_level_index,
                              COHORT_SIZE,
                              n_dle,
                              n_efficate=10)

        # Make sure our model consistently proposes the correct dose level
        dose_escalator.train(False)
        final_proposed_level_index = dose_escalator.propose()
        dose_escalator.train(True)
        if final_proposed_level_index == mtl_index:
            converge_len += 1
        else:
            converge_len = 0
        total_steps += 1

    # Then
    assert total_steps < MAX_ITERATIONS
