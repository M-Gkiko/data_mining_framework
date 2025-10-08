"""
Benchmark core system with Pydantic configuration and streamlined execution.
"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from core.pipeline import Pipeline
from implementations.pipelines import DRAdapter, ClusteringAdapter, DRQualityAdapter, ClusteringQualityAdapter
from core.dataset import Dataset
from core.distance_measure import DistanceMeasure

from .registry import create_algorithm, create_adapter, create_distance_measure


class BenchmarkConfig(BaseModel):
    """ benchmark configuration with automatic validation."""
    
    name: str
    dataset: str = Field(..., description="Path to dataset file")
    iterations: int = Field(default=1, ge=1, description="Number of iterations per combination")
    distance_measure: str = Field(default='Manhattan', description="Distance measure to use")
    output_directory: str = Field(default='benchmark_results', description="Output directory")
    output_formats: List[str] = Field(default=['csv'], description="Output formats")
    pipeline_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline configuration")
    
    class Config:
        extra = "allow"  # Allow additional fields from YAML
    
    @field_validator('dataset')
    def dataset_must_exist(cls, v):
        if not Path(v).exists():
            raise ValueError(f'Dataset file not found: {v}')
        return v
    
    @field_validator('output_formats')
    def validate_formats(cls, v):
        valid = {'csv', 'json', 'yaml'}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f'Invalid formats: {invalid}. Valid: {valid}')
        return v
    
    @property
    def dataset_path(self) -> str:
        """Compatibility property for old code."""
        return self.dataset
    
    def generate_combinations(self) -> List[Dict[str, str]]:
        """Generate all possible algorithm combinations."""
        if not self.pipeline_steps:
            return [{}]
        
        # Extract algorithm options for each step type
        step_types = []
        algorithm_options = []
        
        for step in self.pipeline_steps:
            step_types.append(step['type'])
            algorithm_options.append(step['algorithms'])
        
        # Generate all combinations using itertools.product
        combinations = []
        for combo in product(*algorithm_options):
            combination_dict = dict(zip(step_types, combo))
            combinations.append(combination_dict)
        
        return combinations


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""    
    combination: Dict[str, str]
    execution_time: float
    success: bool
    iteration: int = 0
    error_message: Optional[str] = None
    quality_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = {}
    
    @property
    def is_successful(self) -> bool:
        return self.success and self.error_message is None
    
    @property
    def pipeline_name(self) -> str:
        return "_".join(self.combination.values())


def build_benchmark_pipeline(config: BenchmarkConfig, combination: Dict[str, str], 
                           distance_measure: Optional[DistanceMeasure] = None,
                           dataset: Optional[Dataset] = None) -> Pipeline:
    """Build pipeline from algorithm combination."""
    pipeline_name = "_".join(combination.values())
    pipeline = Pipeline(pipeline_name)
    
    for step in config.pipeline_steps:
        step_type = step['type']
        algorithm_name = combination[step_type]
        params = step.get('params', {}).get(algorithm_name, {})
        
        # Pass distance measure to clustering algorithms that need it
        if step_type == "clustering" and distance_measure is not None:
            params['distance_measure'] = distance_measure
        
        # Create algorithm and adapter
        algorithm = create_algorithm(step_type, algorithm_name, **params)
        adapter = create_adapter(step_type, algorithm, distance_measure, dataset)
        pipeline.add_component(adapter)
    
    return pipeline


class SimpleBenchmark:
    """ Benchmark runner with streamlined execution."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize simplified benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        # Use registry to create distance measure - same pattern as algorithms!
        self.distance_measure = create_distance_measure(config.distance_measure)
        self.results: List[BenchmarkResult] = []
    
    def run(self, dataset: Dataset) -> List[BenchmarkResult]:
        """
        Run all benchmark combinations.
        
        Args:
            dataset: Dataset to run benchmarks on
            
        Returns:
            List of benchmark results
        """
        print(f"Starting benchmark: {self.config.name}")
        print(f"Dataset: {dataset.name}")
        
        combinations = self.config.generate_combinations()
        total_runs = len(combinations) * self.config.iterations
        
        print(f"Testing {len(combinations)} algorithm combinations")
        print(f"Running {self.config.iterations} iterations each")
        print(f"Total benchmark runs: {total_runs}")
        
        for i, combo in enumerate(combinations, 1):
            combo_str = " + ".join([f"{k}:{v}" for k, v in combo.items()])
            print(f"\n--- Combination {i}/{len(combinations)}: {combo} ---")
            
            for iteration in range(self.config.iterations):
                print(f"  Iteration {iteration + 1}/{self.config.iterations}")
                result = self._run_single_benchmark(combo, dataset, iteration + 1)
                self.results.append(result)
                
                if result.is_successful:
                    print(f"    ✓ Success - Total time: {result.execution_time:.3f}s")
                    if result.quality_scores:
                        for metric, score in result.quality_scores.items():
                            print(f"    ✓ Quality scores: {metric}: {score:.3f}")
                else:
                    print(f"    ✗ Failed: {result.error_message}")
        
        successful_runs = len([r for r in self.results if r.is_successful])
        print(f"\nBenchmark completed: {successful_runs}/{len(self.results)} successful")
        
        return self.results
    
    def _run_single_benchmark(self, combo: Dict[str, str], dataset: Dataset, iteration: int) -> BenchmarkResult:
        """Run single benchmark iteration."""
        try:
            pipeline = build_benchmark_pipeline(self.config, combo, self.distance_measure, dataset)
            
            start_time = time.time()
            output = pipeline.execute(dataset)
            execution_time = time.time() - start_time
            
            # Extract quality scores from pipeline output
            quality_scores = self._extract_quality_scores(output)
            
            return BenchmarkResult(
                combination=combo,
                execution_time=execution_time,
                success=True,
                iteration=iteration,
                quality_scores=quality_scores
            )
            
        except Exception as e:
            return BenchmarkResult(
                combination=combo,
                execution_time=0.0,
                success=False,
                iteration=iteration,
                error_message=str(e)
            )
    
    def _extract_quality_scores(self, pipeline_output) -> Dict[str, float]:
        """Extract quality scores from pipeline output."""
        quality_scores = {}
        
        # Handle different output formats
        if isinstance(pipeline_output, dict):
            # Look for quality scores in the output
            for key, value in pipeline_output.items():
                # Check for various quality score key patterns
                if (key.startswith('ClusteringQuality_') or 
                    key.startswith('DRQuality_') or
                    key.startswith('clustering_quality_') or
                    key.startswith('dr_quality_') or
                    'quality' in key.lower()):
                    if isinstance(value, (int, float)):
                        quality_scores[key] = float(value)
                elif key == 'quality_scores' and isinstance(value, dict):
                    quality_scores.update(value)
        
        return quality_scores
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.is_successful]
        
        summary = {
            'total_runs': len(self.results),
            'successful_runs': len(successful_results),
            'failed_runs': len(self.results) - len(successful_results),
            'success_rate': len(successful_results) / len(self.results) * 100 if self.results else 0
        }
        
        if successful_results:
            times = [r.execution_time for r in successful_results]
            summary.update({
                'avg_execution_time': sum(times) / len(times),
                'min_execution_time': min(times),
                'max_execution_time': max(times),
                'total_benchmark_time': sum(times)
            })
            
            # Find fastest and best quality combinations
            fastest = min(successful_results, key=lambda x: x.execution_time)
            summary['fastest_combination'] = {
                'combination': fastest.combination,
                'time': fastest.execution_time,
                'quality_scores': fastest.quality_scores
            }
            
            # Find best quality (highest Calinski-Harabasz or lowest Davies-Bouldin)
            best_quality = None
            for result in successful_results:
                if result.quality_scores:
                    # Prefer Calinski-Harabasz (higher is better), then Davies-Bouldin (lower is better)
                    if any('Calinski' in key for key in result.quality_scores.keys()):
                        if best_quality is None:
                            best_quality = result
                        else:
                            current_score = max([v for k, v in result.quality_scores.items() if 'Calinski' in k])
                            best_score = max([v for k, v in best_quality.quality_scores.items() if 'Calinski' in k])
                            if current_score > best_score:
                                best_quality = result
            
            if best_quality:
                summary['best_quality_combination'] = {
                    'combination': best_quality.combination,
                    'time': best_quality.execution_time,
                    'quality_scores': best_quality.quality_scores
                }
        
        return summary