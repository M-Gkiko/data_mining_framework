"""Pipeline adapter for Edge Measure components."""

from typing import Any, Dict, Tuple
from ...core.pipeline import PipelineComponent
from ...core.edge_measure import EdgeMeasure
from ...core.network import Network


class EdgeMeasureAdapter(PipelineComponent):
    """
    Adapter that wraps Edge Measure algorithms for pipeline execution.

    Handles Network inputs and returns edge measure results.
    """

    def __init__(self, edge_measure: EdgeMeasure, name: str = None, **kwargs):
        """
        Initialize edge measure adapter.

        Args:
            edge_measure: The edge measure to wrap
            name: Name for this component in the pipeline
            **kwargs: Algorithm-specific parameters
        """
        self.edge_measure = edge_measure
        self.algorithm_params = kwargs

        component_name = name or f"EdgeMeasure_{edge_measure.__class__.__name__}"
        super().__init__(component_name)

    def execute(self, input_data) -> Dict[str, Any]:
        """
        Execute edge measure calculation.

        Args:
            input_data: Network object or dict containing 'network' key

        Returns:
            Dict containing:
                - 'edge_scores': Dictionary mapping edge tuples to scores
                - 'measure_name': Name of the measure
                - 'network': Network object used
                - Previous pipeline results (if input was dict)

        Raises:
            ValueError: If input data doesn't contain a Network
            RuntimeError: If calculation fails
        """
        # Extract network from input (handles both Network objects and dicts from previous pipeline steps)
        if isinstance(input_data, Network):
            network = input_data
            previous_results = {}
        elif isinstance(input_data, dict) and 'network' in input_data:
            network = input_data['network']
            previous_results = {k: v for k, v in input_data.items() if k != 'network'}
        else:
            raise ValueError(f"Expected Network or dict with 'network' key, got {type(input_data)}")

        try:
            edge_scores = self.edge_measure.calculate(network, **self.algorithm_params)
        except Exception as e:
            raise RuntimeError(f"Edge measure calculation failed: {str(e)}") from e

        result = {
            **previous_results,  # Include results from previous pipeline steps
            'edge_scores': edge_scores,
            'measure_name': self.edge_measure.__class__.__name__,
            'network': network
        }
        return result
