from abc import abstractmethod, ABC


class DoseEscalatorBase(ABC):
    """
    Base class for dose escalators.
    """

    @abstractmethod
    def propose(self) -> int:
        """
        Propose the index of the next dose to trial.
        """

    @abstractmethod
    def update(self, dose_level_index: int, cohort_size: int, n_dle: int):
        """
        Update the escalator with the environment feedback.
        """
