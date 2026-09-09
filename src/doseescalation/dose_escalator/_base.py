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

    @property
    def stopped(self) -> bool:
        """
        Whether the design has terminated the trial and enrols no further
        patients (e.g. 3 + 3 once it declares an MTD). Model-based designs that
        keep allocating for the whole horizon (CRM, UCB, SEEDA, ...) return
        False, which is the default here.
        """
        return False
