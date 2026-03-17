import math
import numpy as np
import pytest
from doseescalation.estimator import PowerEstimator


@pytest.mark.unit
def test_power_instantiation():
    PowerEstimator(
        mu=1.0, sigma=2.0, num_results=10, num_burnin_steps=10
    )


@pytest.mark.unit
def test_power_fit():
    estimator = PowerEstimator(
        mu=1.0, sigma=1.0, num_results=10, num_burnin_steps=10
    )
    estimator.fit(((200, 10), (300, 20)), (100, 60))


@pytest.mark.unit
def test_power_predict():
    estimator = PowerEstimator(
        mu=1.0, sigma=1.0, num_results=10, num_burnin_steps=10
    )
    estimator.predict((0.1, 0.2, 0.6))


@pytest.mark.unit
def test_power_expectation():
    estimator = PowerEstimator(
        mu=1.0, sigma=1.0, num_results=10, num_burnin_steps=10
    )
    estimator.expectation((0.1, 0.2, 0.6), lambda x: x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "beta, cohort_doses",
    [
        (-2.0, ((100, 0.1), (100, 0.2), (200, 0.3))),
        (1.0, ((100, 0.5), (200, 0.3), (300, 0.1))),
    ]
)
def test_power_posterior_closer_to_obs(beta, cohort_doses):
    # Given
    estimator = PowerEstimator(
        num_results=200, num_burnin_steps=200
    )
    n_dles = [
        round((dose ** math.exp(beta)) * cohort)
        for cohort, dose in cohort_doses
    ]
    p_dles = [
        n_dles[idx] / cohort_doses[idx][0]
        for idx in range(len(n_dles))
    ]
    doses = [x[1] for x in cohort_doses]
    initial_preds = estimator.predict(doses)

    # When
    for _ in range(10):
        estimator.fit(cohort_doses, n_dles)

    # Then
    fitted_preds = estimator.predict(doses)
    assert (
        np.linalg.norm(fitted_preds - p_dles)
        < np.linalg.norm(initial_preds - p_dles)
    )
