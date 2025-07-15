#!/usr/bin/env python3
"""
Configuration Loader for AFPM Analysis Framework
Reads and validates configuration from YAML file
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

class AFPMConfig:
    """Configuration manager for AFPM analysis framework"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_file: Path to YAML configuration file
        """
        # If config_file is just a filename, look for it in the same directory as this script
        if Path(config_file).is_absolute():
            self.config_file = Path(config_file)
        else:
            # Get directory of this script
            script_dir = Path(__file__).parent
            self.config_file = script_dir / config_file
            
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate matrix generation settings
        matrix_gen = self.config.get('matrix_generation', {})
        size_range = matrix_gen.get('size_range', {})
        
        min_size = size_range.get('min_size', 4)
        max_size = size_range.get('max_size', 65536)
        
        if min_size < 4:
            raise ValueError("min_size must be at least 4")
        if max_size < min_size:
            raise ValueError("max_size must be greater than min_size")
        
        # Validate AFPM configurations
        afpm_configs = self.config.get('afpm_analysis', {}).get('configurations', {})
        for name, config in afpm_configs.items():
            chromosome = config.get('chromosome', [])
            if len(chromosome) != 9:
                raise ValueError(f"Configuration '{name}' chromosome must have 9 elements")
            
            for i, val in enumerate(chromosome):
                if val != -1 and not (0 <= val <= 99):
                    raise ValueError(f"Configuration '{name}' chromosome[{i}] must be -1 or 0-99")
    
    def get_matrix_sizes(self) -> List[int]:
        """Get list of matrix sizes based on configuration"""
        size_range = self.config['matrix_generation']['size_range']
        min_size = size_range['min_size']
        max_size = size_range['max_size']
        progression = size_range['size_progression']
        
        if progression == "powers_of_2":
            # Generate powers of 2 from min_size to max_size
            sizes = []
            
            # Start from the smallest power of 2 >= min_size
            if min_size <= 1:
                current_power = 0
            else:
                current_power = int(np.ceil(np.log2(min_size)))
            
            while True:
                size = 2 ** current_power
                if size > max_size:
                    break
                if size >= min_size:
                    sizes.append(size)
                current_power += 1
                
                # Safety check to prevent infinite loops
                if current_power > 20:  # 2^20 = 1M, should be enough
                    break
            
            return sizes
        
        elif progression == "linear":
            step_size = size_range['step_size']
            return list(range(min_size, max_size + 1, step_size))
        
        elif progression == "custom":
            custom_sizes = size_range['custom_sizes']
            # Filter sizes within min/max range
            return [s for s in custom_sizes if min_size <= s <= max_size]
        
        else:
            raise ValueError(f"Unknown size_progression: {progression}")
    
    def get_matrix_types(self) -> List[str]:
        """Get list of matrix types to generate"""
        return self.config['matrix_generation']['matrix_types']
    
    def get_matrices_per_size(self) -> int:
        """Get number of matrices to generate per size"""
        return self.config['matrix_generation']['matrices_per_size']
    
    def get_output_dir(self) -> Path:
        """Get output directory for matrices"""
        output_dir = self.config['matrix_generation']['output_dir']
        # If it's a relative path, make it relative to the script directory
        if not Path(output_dir).is_absolute():
            script_dir = Path(__file__).parent
            return script_dir / output_dir
        return Path(output_dir)
    
    def get_results_dir(self) -> Path:
        """Get results directory for analysis outputs"""
        results_dir = self.config['afpm_analysis']['results_dir']
        # If it's a relative path, make it relative to the script directory
        if not Path(results_dir).is_absolute():
            script_dir = Path(__file__).parent
            return script_dir / results_dir
        return Path(results_dir)
    
    def get_random_seed(self) -> int:
        """Get random seed for reproducibility"""
        return self.config['matrix_generation']['random_seed']
    
    def get_afpm_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get AFPM configurations"""
        return self.config['afpm_analysis']['configurations']
    
    def get_configuration_names(self) -> List[str]:
        """Get list of configuration names"""
        return list(self.config['afpm_analysis']['configurations'].keys())
    
    def get_chromosome(self, config_name: str) -> List[int]:
        """Get chromosome for specific configuration"""
        configs = self.get_afpm_configurations()
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        return configs[config_name]['chromosome']
    
    def get_configuration_color(self, config_name: str) -> str:
        """Get color for specific configuration"""
        configs = self.get_afpm_configurations()
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        return configs[config_name].get('color', '#000000')
    
    def get_configuration_description(self, config_name: str) -> str:
        """Get description for specific configuration"""
        configs = self.get_afpm_configurations()
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        return configs[config_name].get('description', 'Custom configuration')
    
    def get_error_metrics(self) -> List[str]:
        """Get list of error metrics to compute"""
        return self.config['afpm_analysis']['error_metrics']
    
    def get_float32_epsilon(self) -> float:
        """Get float32 epsilon for theoretical bounds"""
        return self.config['afpm_analysis']['theoretical_bounds']['float32_epsilon']
    
    def get_thomas_constant(self) -> float:
        """Get Thomas algorithm constant"""
        return self.config['afpm_analysis']['theoretical_bounds']['thomas_constant']
    
    def get_plot_types(self) -> List[str]:
        """Get list of plot types to generate"""
        return self.config['plotting']['plot_types']
    
    def get_figure_size(self) -> tuple:
        """Get figure size for plots"""
        size = self.config['plotting']['figure_size']
        return (size[0], size[1])
    
    def get_dpi(self) -> int:
        """Get DPI for saved figures"""
        return self.config['plotting']['dpi']
    
    def should_show_error_bars(self) -> bool:
        """Check if error bars should be shown"""
        return self.config['plotting']['error_bars']
    
    def should_show_grid(self) -> bool:
        """Check if grid should be shown"""
        return self.config['plotting']['grid']
    
    def get_legend_fontsize(self) -> int:
        """Get legend font size"""
        return self.config['plotting']['legend_fontsize']
    
    def get_grid_alpha(self) -> float:
        """Get grid transparency"""
        return self.config['plotting']['grid_alpha']
    
    def is_parallel_enabled(self) -> bool:
        """Check if parallel processing is enabled"""
        return self.config['advanced']['enable_parallel']
    
    def get_num_processes(self) -> int:
        """Get number of parallel processes"""
        return self.config['advanced']['num_processes']
    
    def get_pivot_threshold(self) -> float:
        """Get pivot detection threshold"""
        return self.config['advanced']['pivot_threshold']
    
    def should_skip_failed_matrices(self) -> bool:
        """Check if failed matrices should be skipped"""
        return self.config['advanced']['skip_failed_matrices']
    
    def is_verbose_output(self) -> bool:
        """Check if verbose output is enabled"""
        return self.config['advanced']['verbose_output']
    
    def get_max_condition_number(self) -> float:
        """Get maximum allowed condition number"""
        return self.config['validation']['max_condition_number']
    
    def get_max_residual(self) -> float:
        """Get maximum acceptable residual"""
        return self.config['validation']['max_residual']
    
    def should_check_condition_number(self) -> bool:
        """Check if condition number validation is enabled"""
        return self.config['validation']['check_condition_number']
    
    def should_compare_with_exact(self) -> bool:
        """Check if comparison with exact solver is enabled"""
        return self.config['validation']['compare_with_exact']
    
    def print_summary(self):
        """Print configuration summary"""
        print("=== AFPM Configuration Summary ===")
        print(f"Config file: {self.config_file}")
        print()
        
        print("Matrix Generation:")
        sizes = self.get_matrix_sizes()
        print(f"  Sizes: {len(sizes)} sizes from {min(sizes)} to {max(sizes)}")
        print(f"  Types: {', '.join(self.get_matrix_types())}")
        print(f"  Matrices per size: {self.get_matrices_per_size()}")
        print(f"  Total matrices: {len(sizes) * len(self.get_matrix_types()) * self.get_matrices_per_size()}")
        print(f"  Output directory: {self.get_output_dir()}")
        print()
        
        print("AFPM Analysis:")
        configs = self.get_configuration_names()
        print(f"  Configurations: {len(configs)} ({', '.join(configs)})")
        print(f"  Error metrics: {', '.join(self.get_error_metrics())}")
        print(f"  Results directory: {self.get_results_dir()}")
        print()
        
        print("Plotting:")
        plot_types = self.get_plot_types()
        print(f"  Plot types: {', '.join(plot_types)}")
        print(f"  Figure size: {self.get_figure_size()}")
        print(f"  DPI: {self.get_dpi()}")
        print()
        
        print("Advanced:")
        print(f"  Parallel processing: {self.is_parallel_enabled()}")
        print(f"  Verbose output: {self.is_verbose_output()}")
        max_cond = self.get_max_condition_number()
        print(f"  Max condition number: {float(max_cond):.1e}")


def load_config(config_file: str = "config.yaml") -> AFPMConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        AFPMConfig object
    """
    return AFPMConfig(config_file)


def main():
    """Test configuration loading"""
    try:
        config = load_config()
        config.print_summary()
        
        print("\n=== Configuration Details ===")
        print("Matrix sizes:", config.get_matrix_sizes())
        print()
        
        print("AFPM Configurations:")
        for name in config.get_configuration_names():
            chromosome = config.get_chromosome(name)
            description = config.get_configuration_description(name)
            color = config.get_configuration_color(name)
            print(f"  {name}: {chromosome}")
            print(f"    Description: {description}")
            print(f"    Color: {color}")
            print()
        
    except Exception as e:
        print(f"Error loading configuration: {e}")


if __name__ == "__main__":
    main()