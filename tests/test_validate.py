import random
import pytest
from doseescalation.dose_escalator import NoOpValidator, NoSkipValidator


@pytest.mark.unit
def test_no_op_validator():
    # Given
    no_op = NoOpValidator()

    # When & Then
    for _ in range(10):
        no_op.visit(random.randint(0, 10))
        assert no_op.validate(random.randint(0, 10))


@pytest.mark.unit
def test_no_skip_validator():
    # Given
    no_skip = NoSkipValidator(3)
    assert not no_skip.validate(1)
    assert not no_skip.validate(2)

    # When
    no_skip.visit(0)

    # Then
    assert no_skip.validate(1)
    assert not no_skip.validate(2)
