"""Pipeline adapter for Node Measure components."""

from typing import Any, Dict
from ...core.pipeline import PipelineComponent
from ...core.node_measure import NodeMeasure
from ...core.network import Network


class NodeMeasureAdapter(PipelineComponent):
    """
    Adapter that wraps Node Measure algorithms for pipeline execution.

    Handles Network inputs and returns node measure results.
    """

    def __init__(self, node_measure: NodeMeasure, name: str = None, **kwargs):
        """
        Initialize node measure adapter.

        Args:
            node_measure: The node measure to wrap
            name: Name for this component in the pipeline
            **kwargs: Algorithm-specific parameters
        """
        self.node_measure = node_measure
        self.algorithm_params = kwargs

        component_name = name or f"NodeMeasure_{node_measure.__class__.__name__}"
        super().__init__(component_name)

    def execute(self, input_data) -> Dict[str, Any]:
        """
        Execute node measure calculation.

        Args:
            input_data: Network object or dict containing 'network' key

        Returns:
            Dict containing:
                - 'node_scores': Dictionary mapping nodes to scores
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
            node_scores = self.node_measure.calculate(network, **self.algorithm_params)
        except Exception as e:
            raise RuntimeError(f"Node measure calculation failed: {str(e)}") from e

        result = {
            **previous_results,  # Include results from previous pipeline steps
            'node_scores': node_scores,
            'measure_name': self.node_measure.__class__.__name__,
            'network': network
        }
        return result
