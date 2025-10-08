"""
Configuration loader for benchmark YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from .benchmark import BenchmarkConfiguration


class BenchmarkConfigLoader:
    """Loads and parses benchmark configuration from YAML files."""
    
    @staticmethod
    def load_config(config_path: str) -> BenchmarkConfiguration:
        """
        Load benchmark configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            BenchmarkConfiguration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If required fields are missing
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        return BenchmarkConfigLoader._parse_config(yaml_data)
    
    @staticmethod
    def _parse_config(yaml_data: Dict[str, Any]) -> BenchmarkConfiguration:
        """Parse YAML data into BenchmarkConfiguration."""
        
        # Validate required fields
        if 'benchmark' not in yaml_data:
            raise ValueError("Missing 'benchmark' section in config")
        
        benchmark_section = yaml_data['benchmark']
        
        required_fields = ['name', 'dataset']
        for field in required_fields:
            if field not in benchmark_section:
                raise ValueError(f"Missing required field 'benchmark.{field}' in config")
        
        # Create configuration
        config = BenchmarkConfiguration(
            name=benchmark_section['name'],
            dataset_path=benchmark_section['dataset'],
            iterations=yaml_data.get('iterations', 1),
            output_directory=yaml_data.get('output', {}).get('directory', 'benchmark_results'),
            output_formats=yaml_data.get('output', {}).get('format', ['csv']),
            distance_measure=yaml_data.get('distance_measure', 'Manhattan')
        )
        
        # Parse pipeline template
        if 'pipeline_template' in yaml_data:
            for step in yaml_data['pipeline_template']:
                if 'type' not in step or 'algorithms' not in step:
                    raise ValueError("Each pipeline step must have 'type' and 'algorithms' fields")
                
                config.add_pipeline_step(
                    step_type=step['type'],
                    algorithms=step['algorithms'],
                    params=step.get('params', {})
                )
        
        return config
    
    @staticmethod
    def validate_config(config: BenchmarkConfiguration) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check dataset path
        if not Path(config.dataset_path).exists():
            issues.append(f"Dataset file not found: {config.dataset_path}")
        
        # Check pipeline steps
        if not config.pipeline_steps:
            issues.append("No pipeline steps defined")
        
        # Check iterations
        if config.iterations < 1:
            issues.append("Iterations must be >= 1")
        
        # Check output formats
        valid_formats = ['csv', 'json', 'yaml']
        for fmt in config.output_formats:
            if fmt not in valid_formats:
                issues.append(f"Unsupported output format: {fmt}. Valid: {valid_formats}")
        
        return issues