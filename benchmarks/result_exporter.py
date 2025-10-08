"""
Export benchmark results to various formats.
"""

import csv
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .benchmark import BenchmarkResult


class BenchmarkResultExporter:
    """Exports benchmark results to different file formats."""
    
    @staticmethod
    def export_results(results: List[BenchmarkResult], 
                      output_directory: str,
                      formats: List[str],
                      benchmark_name: str) -> None:
        """
        Export benchmark results to specified formats.
        
        Args:
            results: List of benchmark results
            output_directory: Directory to save results
            formats: List of output formats ('csv', 'json', 'yaml')
            benchmark_name: Name for output files
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{benchmark_name}_{timestamp}"
        
        for format_type in formats:
            if format_type == 'csv':
                BenchmarkResultExporter._export_csv(
                    results, output_dir / f"{base_filename}.csv"
                )
            elif format_type == 'json':
                BenchmarkResultExporter._export_json(
                    results, output_dir / f"{base_filename}.json"
                )
            elif format_type == 'yaml':
                BenchmarkResultExporter._export_yaml(
                    results, output_dir / f"{base_filename}.yaml"
                )
    
    @staticmethod
    def _export_csv(results: List[BenchmarkResult], output_path: Path) -> None:
        """Export results to CSV format."""
        if not results:
            return
        
        # Determine all possible columns
        fieldnames = ['pipeline_name', 'iteration', 'success', 'total_time', 'error_message']
        
        # Add algorithm combination columns
        all_algo_types = set()
        for result in results:
            all_algo_types.update(result.algorithm_combination.keys())
        
        for algo_type in sorted(all_algo_types):
            fieldnames.append(f"algorithm_{algo_type}")
        
        # Add execution time columns
        all_components = set()
        for result in results:
            all_components.update(result.execution_times.keys())
        
        for component in sorted(all_components):
            fieldnames.append(f"time_{component}")
        
        # Add quality score columns
        all_metrics = set()
        for result in results:
            all_metrics.update(result.quality_scores.keys())
        
        for metric in sorted(all_metrics):
            fieldnames.append(f"quality_{metric}")
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'pipeline_name': result.pipeline_name,
                    'iteration': result.iteration,
                    'success': result.success,
                    'total_time': result.total_time,
                    'error_message': result.error_message or ''
                }
                
                # Add algorithm combinations
                for algo_type in all_algo_types:
                    row[f"algorithm_{algo_type}"] = result.algorithm_combination.get(algo_type, '')
                
                # Add execution times
                for component in all_components:
                    row[f"time_{component}"] = result.execution_times.get(component, '')
                
                # Add quality scores
                for metric in all_metrics:
                    row[f"quality_{metric}"] = result.quality_scores.get(metric, '')
                
                writer.writerow(row)
        
        print(f"Results exported to CSV: {output_path}")
    
    @staticmethod
    def _export_json(results: List[BenchmarkResult], output_path: Path) -> None:
        """Export results to JSON format."""
        json_data = {
            'benchmark_info': {
                'export_timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'successful_results': len([r for r in results if r.success])
            },
            'results': []
        }
        
        for result in results:
            result_dict = {
                'pipeline_name': result.pipeline_name,
                'iteration': result.iteration,
                'algorithm_combination': result.algorithm_combination,
                'execution_times': result.execution_times,
                'total_time': result.total_time,
                'quality_scores': result.quality_scores,
                'success': result.success,
                'error_message': result.error_message
            }
            json_data['results'].append(result_dict)
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Results exported to JSON: {output_path}")
    
    @staticmethod
    def _export_yaml(results: List[BenchmarkResult], output_path: Path) -> None:
        """Export results to YAML format."""
        yaml_data = {
            'benchmark_info': {
                'export_timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'successful_results': len([r for r in results if r.success])
            },
            'results': []
        }
        
        for result in results:
            result_dict = {
                'pipeline_name': result.pipeline_name,
                'iteration': result.iteration,
                'algorithm_combination': result.algorithm_combination,
                'execution_times': result.execution_times,
                'total_time': result.total_time,
                'quality_scores': result.quality_scores,
                'success': result.success,
                'error_message': result.error_message
            }
            yaml_data['results'].append(result_dict)
        
        with open(output_path, 'w', encoding='utf-8') as yamlfile:
            yaml.dump(yaml_data, yamlfile, default_flow_style=False, allow_unicode=True)
        
        print(f"Results exported to YAML: {output_path}")