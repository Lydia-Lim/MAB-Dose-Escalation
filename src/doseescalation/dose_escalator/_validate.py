import abc


class Validator(abc.ABC):
    @abc.abstractmethod
    def visit(self, dose_level_index: int):
        """
        Record a newly visited dose level.
        """

    @abc.abstractmethod
    def validate(self, dose_level_index: int):
        """
        Validate the target dose level.
        """


class NoOpValidator(Validator):
    def visit(self, _: int):
        pass

    def validate(self, _: int):
        return True


class NoSkipValidator(Validator):
    """
    A validator to make sure our dose escalator never skips any dose level.
    """

    def __init__(self, n_dose_levels):
        self._n_dose_levels = n_dose_levels
        self._visited_dose_levels = set()

    def visit(self, dose_level_index: int):
        if dose_level_index not in range(self._n_dose_levels):
            raise ValueError(
                f"dose_level_index={dose_level_index} is out of range"
            )
        self._visited_dose_levels.add(dose_level_index)

    def validate(self, dose_level_index: int) -> bool:
        if dose_level_index not in range(self._n_dose_levels):
            raise ValueError(
                f"dose_level_index={dose_level_index} is out of range"
            )
        if dose_level_index == 0:
            return True
        return dose_level_index - 1 in self._visited_dose_levels
