import numpy as np
import pytest
from doseescalation.estimator import AveragingEstimator


@pytest.mark.unit
def test_avg_instantiation():
    AveragingEstimator()


@pytest.mark.unit
def test_avg_fit():
    estimator = AveragingEstimator()
    estimator.fit(((200, 10), (300, 20)), (100, 60))


@pytest.mark.unit
def test_avg_predict():
    estimator = AveragingEstimator()
    estimator.predict((0.1, 0.2, 0.6))


@pytest.mark.unit
@pytest.mark.parametrize(
    "cohort_doses, n_dles",
    [
        (((100, 0.1), (100, 0.2), (200, 0.3)), (30, 40, 100)),
        (((100, 0.5), (200, 0.3), (300, 0.1)), (20, 20, 20)),
    ]
)
def test_avg_fitted_closer_to_obs(cohort_doses, n_dles):
    # Given
    estimator = AveragingEstimator()
    p_dles = np.array([
        n_dles[idx] / cohort_doses[idx][0]
        for idx in range(len(n_dles))
    ])
    doses = [x[1] for x in cohort_doses]
    initial_preds = estimator.predict(doses)

    # When
    estimator.fit(cohort_doses, n_dles)

    # Then
    fitted_preds = estimator.predict(doses)
    assert (
        np.linalg.norm(np.array(fitted_preds) - p_dles)
        < np.linalg.norm(np.array(initial_preds) - p_dles)
    )
