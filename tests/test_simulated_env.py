import numpy as np
import pytest
from doseescalation.simulated_env import SimulatedEnv
from numbers import Number


@pytest.mark.unit
def test_instantiation():
    SimulatedEnv((), lambda x: x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dose_level_index, cohort_size",
    [(0, 10), (1, 100), (2, 1000)]
)
def test_dose_response(dose_level_index: Number, cohort_size: int):
    # Given
    dose_levels = (3, 5, 7)
    np.random.seed(123)
    dose_toxic_curve = np.tanh

    # When
    dose_response = SimulatedEnv(dose_levels, dose_toxic_curve)
    n_dle = dose_response(dose_level_index, cohort_size)

    # Then
    assert isinstance(n_dle, int)
    dose = dose_levels[dose_level_index]
    pytest.approx(n_dle, 0.1) == dose_toxic_curve(dose) * cohort_size


@pytest.mark.unit
def test_non_parametric_env():
    """
    Dose levels and toxicity/efficacy probabilities follow the
    parameterisation set up in the SEEDA paper "Learning for
    Dose Allocation in Adaptive Clinical Trials with Safety
    Constraints".

    The generic set up of the SimulatedEnv() class allows us
    to arbitrarily define any mapping between a dose level set
    of indexes, and toxicity/efficacy.
    """
    cohort_size = 10000
    dose_levels = np.array([0, 1, 2, 3, 4, 5])
    toxicity_p_vals = np.array([0.01, 0.05, 0.15, 0.2, 0.45, 0.6])
    efficacy_q_vals = np.array([0.1, 0.35, 0.6, 0.6, 0.6, 0.6])

    assert len(dose_levels) == len(toxicity_p_vals)
    assert len(dose_levels) == len(efficacy_q_vals)

    def non_parametric_dose_toxic_curve(dose_levels_indices, p_vals):
        """
        Extracts the values in p_vals corresponding with the
        indices provided in dose_level_indices.
        """
        return p_vals[dose_levels_indices]

    toxicity_dose_response = SimulatedEnv(
        dose_levels,
        lambda x: non_parametric_dose_toxic_curve(x, p_vals=toxicity_p_vals)
    )
    efficacy_dose_response = SimulatedEnv(
        dose_levels,
        lambda x: non_parametric_dose_toxic_curve(x, p_vals=efficacy_q_vals)
    )

    np.random.seed(0)
    dose_level_index = np.random.choice(dose_levels)
    n_dle = toxicity_dose_response(dose_level_index, cohort_size)
    n_efficate = efficacy_dose_response(dose_level_index, cohort_size)

    assert isinstance(n_dle, int)
    assert isinstance(n_efficate, int)
    dose = dose_levels[dose_level_index]
    pytest.approx(n_dle, 0.1) == (
        non_parametric_dose_toxic_curve(dose, toxicity_p_vals) * cohort_size
    )
    pytest.approx(n_efficate, 0.1) == (
        non_parametric_dose_toxic_curve(dose, efficacy_q_vals) * cohort_size
    )
