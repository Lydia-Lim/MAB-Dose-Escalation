import numpy as np
import pytest
from doseescalation.estimator import TanhIntegrativeEstimator


@pytest.mark.unit
def test_non_tf_tanh_instantiation():
    TanhIntegrativeEstimator(
        prior_scale=4, prior_rate=4
    )


@pytest.mark.unit
def test_non_tf_tanh_fit():
    estimator = TanhIntegrativeEstimator(
        prior_scale=1, prior_rate=1
    )
    estimator.fit(((200, 0.01), (300, 0.1)), (100, 60))


@pytest.mark.unit
def test_non_tf_tanh_predict():
    estimator = TanhIntegrativeEstimator(
        prior_scale=4, prior_rate=4
    )
    estimator.predict((0.1, 0.2, 0.6))


@pytest.mark.unit
@pytest.mark.parametrize(
    "beta, cohort_doses",
    [
        (2.0, ((100, 0.1), (100, 0.2), (200, 0.3))),
        (1.1, ((100, 0.5), (200, 0.3), (300, 0.1))),
    ]
)
def test_non_tf_tanh_posterior_closer_to_obs(beta, cohort_doses):
    # Given
    estimator = TanhIntegrativeEstimator(
        prior_scale=4, prior_rate=1
    )
    n_dles = [
        round(((
            (np.tanh(dose) + 1) / 2
        ) ** beta) * cohort)
        for cohort, dose in cohort_doses
    ]
    p_dles = [
        n_dles[idx] / cohort_doses[idx][0]
        for idx in range(len(n_dles))
    ]
    doses = [x[1] for x in cohort_doses]

    # Initial prediction generates estimates from `a` prior
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
