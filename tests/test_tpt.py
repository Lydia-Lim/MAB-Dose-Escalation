import pytest
from doseescalation.dose_escalator import ThreePlusThreeDoseEscalator


@pytest.mark.unit
def test_stage_0_lowest_dose_level():
    # Given
    dose_levels = [10, 20, 30]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # When
    proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 0


@pytest.mark.unit
def test_stage_0_update_n0():
    # Given
    dose_levels = [10, 20, 30]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)

    # When
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 0
    # step 3 - send the feedback to the escalator to update it
    dose_escalator.update(proposed_dose_index, 3, n_dle)
    # step 4 - ask the escalator for another proposal
    proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 1


@pytest.mark.unit
def test_stage_0_update_n2():
    # Given
    dose_levels = [10, 20, 30]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # When
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    assert proposed_dose_index == 0
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 2
    # step 3 - send the feedback to the escalator to update it
    dose_escalator.update(proposed_dose_index, 3, n_dle)
    # step 4 - ask the escalator for another proposal
    proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 0


@pytest.mark.unit
def test_stage_1_update_n1_n0():
    # Given
    dose_levels = [10, 20, 30]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # When
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 1
    # step 3 - send the feedback to the escalator to update it
    dose_escalator.update(proposed_dose_index, 3, n_dle)
    # step 4 - ask the escalator for another proposal
    proposed_dose_index = dose_escalator.propose()
    # step 5 - send the proposal to the environment and get a feedback again
    n_dle = 0
    # step 6 - send the feedback to the escalator to update it, and
    dose_escalator.update(proposed_dose_index, 3, n_dle)
    # step 7 - ask the escalator for another proposal
    proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 1


@pytest.mark.unit
def test_stage_1_update_n1_n1():
    # Given
    dose_levels = [10, 20, 30]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # When
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 1
    # step 3 - send the feedback to the escalator to update it
    dose_escalator.update(proposed_dose_index, 3, n_dle)
    # step 4 - ask the escalator for another proposal
    proposed_dose_index = dose_escalator.propose()
    # step 5 - send the proposal to the environment and get a feedback again
    n_dle = 1
    # step 6 - send the feedback to the escalator to update it, and
    dose_escalator.update(proposed_dose_index, 3, n_dle)
    # step 7 - ask the escalator for another proposal
    proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 0


@pytest.mark.unit
def test_escalate_stage_0_n0():
    # Given
    dose_levels = [10, 20, 30, 40, 50]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 0
    # step 3 - start loop to test different dosage levels to escalate
    last_index = None
    while proposed_dose_index != last_index:
        last_index = proposed_dose_index
        if proposed_dose_index == 4:
            n_dle = 2
        # step 3.1 - send the feedback to the escalator to update it
        dose_escalator.update(proposed_dose_index, 3, n_dle)
        # step 3.2 - ask the escalator for another proposal
        proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 3


@pytest.mark.unit
def test_escalate_stage_2_n2():
    # Given
    dose_levels = [10, 20, 30, 40, 50]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 0
    # step 3 - start loop to test different dosage levels to escalate
    last_index = None
    stage = 0
    while proposed_dose_index != last_index or stage == 1:
        last_index = proposed_dose_index
        if proposed_dose_index == 2:
            n_dle = 2
            stage = 2
        # step 3.1 - send the feedback to the escalator to update it
        dose_escalator.update(proposed_dose_index, 3, n_dle)
        # step 3.2 - ask the escalator for another proposal
        proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 1


@pytest.mark.unit
def test_escalate_stage_1_n1_n1():
    # Given
    dose_levels = [10, 20, 30, 40, 50]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 0
    # step 3 - start loop to test different dosage levels to escalate
    last_index = None
    stage = 0
    while proposed_dose_index != last_index or stage == 1:
        last_index = proposed_dose_index
        if proposed_dose_index == 2:
            n_dle = 1
            stage = stage + 1
        # step 3.1 - send the feedback to the escalator to update it
        dose_escalator.update(proposed_dose_index, 3, n_dle)
        # step 3.2 - ask the escalator for another proposal
        proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 1


@pytest.mark.unit
def test_escalate_stage_1_n1_n0():
    # Given
    dose_levels = [10, 20, 30, 40, 50]
    dose_escalator = ThreePlusThreeDoseEscalator(dose_levels)
    # step 1 - ask dose escalator for the first proposal
    proposed_dose_index = dose_escalator.propose()
    # step 2 - send the proposal to the environment and get a feedback
    n_dle = 0
    # step 3 - start loop to test different dosage levels to escalate
    last_index = None
    stage = 0
    while proposed_dose_index != last_index or stage == 1:
        last_index = proposed_dose_index
        if proposed_dose_index == 2 and stage == 1:
            n_dle = 0
        if proposed_dose_index == 2 and stage == 0:
            n_dle = 1
            stage = 1
        if proposed_dose_index == 3:
            n_dle = 1
            stage = 2
        # step 3.1 - send the feedback to the escalator to update it
        dose_escalator.update(proposed_dose_index, 3, n_dle)
        # step 3.2 - ask the escalator for another proposal
        proposed_dose_index = dose_escalator.propose()
    # Then
    assert proposed_dose_index == 2
