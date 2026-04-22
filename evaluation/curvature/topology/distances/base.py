from abc import ABC, abstractmethod
import numpy as np


class TopologicalDistance(ABC):
    """Abstract base class for computing topological distances.

    Takes in two persistence diagrams (or lists of persistence diagrams) and handles necessary transformations
    to compute a specific distance function.

    Attributes
    ----------
    diagram 1 : List[PersistenceDiagram]
        The List[PersistenceDiagram] to be compared with diagram2.
    diagram 2 : List[PersistenceDiagram]
        The List[PersistenceDiagram] to be compared with diagram1.
    norm : int, default=2.
        Defines what norm will be used for calculations. Default is 2.
    """

    def __init__(self, diagram1, diagram2, norm=2) -> None:
        """
        Initializes an instance of the TopologicalDistance class, and converts diagram1 and diagram2 to lists if they are not already.

        Parameters
        ----------
        diagram1 : PersistenceDiagram or List[PersistenceDiagram]
            The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram2.
        diagram 2 : PersistenceDiagram of List[PersistenceDiagram]
            The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram1.
        norm : int, default=2.
            Defines what norm will be used for calculations. Default is 2.
        Returns
        -------
        None
        """
        super().__init__()
        self.norm_order = norm

        # Ensure support for distributions if provided
        if self._is_distribution(diagram1) or self._is_distribution(diagram2):
            assert (
                self.supports_distribution()
            ), "Distribution of persistence diagrams is not supported by this distance type."

        self.diagram1 = diagram1 if isinstance(diagram1, list) else [diagram1]
        self.diagram2 = diagram2 if isinstance(diagram2, list) else [diagram2]

    def compute_norm(self, x):
        """Compute norm of a vector x according to the specified order."""
        return np.linalg.norm(x, ord=self.norm_order)

    @staticmethod
    def _is_distribution(diagram):
        """Check if the given diagram is a distribution (i.e., a list of diagrams)."""
        return isinstance(diagram, list)

    @abstractmethod
    def supports_distribution(self) -> bool:
        """Indicates if the distance type supports distributions of persistence diagrams."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs):
        """Converts persistence diagrams to topological descriptors."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, descriptor1, descriptor2) -> float:
        """Defines how to compute distances between topological descriptors."""
        raise NotImplementedError
