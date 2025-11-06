from abc import ABC, abstractmethod
from typing import Dict, Any
from .network import Network


class NodeMeasure(ABC):
    """
    Abstract base class for node-level measures in networks.

    This interface defines the contract that all node measure implementations
    must follow, enabling the Strategy Pattern for node analysis operations.
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the node measure with configuration parameters.

        Args:
            **kwargs: Algorithm-specific hyperparameters
        """
        pass

    @abstractmethod
    def calculate(self, network: Network, **kwargs: Any) -> Dict[Any, float]:
        """
        Calculate the measure for all nodes in the network.

        Args:
            network (Network): The network to analyze
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dict[Any, float]: Dictionary mapping node identifiers to their measure values

        Raises:
            ValueError: If network is invalid or empty
        """
        pass
