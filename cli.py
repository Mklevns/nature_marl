# File: nature_marl/cli.py
"""
Command Line Interface for Nature-Inspired Multi-Agent Reinforcement Learning

This CLI provides easy access to training, testing, benchmarking, and analysis
tools for the bio-inspired MARL system.

FEATURES:
✅ Interactive training with real-time monitoring
✅ System compatibility testing
✅ Performance benchmarking
✅ Bio-inspired metrics analysis
✅ Model export and conversion
✅ Research experiment management
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.panel import Panel
from rich.tree import Tree

# Import Nature MARL components
from nature_marl.core.production_bio_inspired_rl_module import (
    ProductionUnifiedBioInspiredRLModule,
    create_production_bio_module_spec
)
from nature_marl.utils.logging_config import (
    setup_production_logging,
    BioInspiredMetricsTracker,
    bio_inspired_logging_context
)
from nature_marl.utils.debug_utils import (
    run_system_tests,
    print_system_info,
    benchmark_performance
)

# Initialize rich console
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("nature_marl.cli")


class NatureMARL_CLI:
    """Main CLI application for Nature MARL."""

    def __init__(self):
        self.console = console
        self.version = "1.0.0"

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="nature-marl",
            description="🌿 Nature-Inspired Multi-Agent Reinforcement Learning CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  nature-marl train --config ant_colony.yaml --iterations 100
  nature-marl test --all --verbose
  nature-marl benchmark --agents 8 --hidden-dim 512
  nature-marl analyze results/experiment_1/metrics.json
  nature-marl export model.pt --format onnx
            """
        )

        parser.add_argument(
            "--version",
            action="version",
            version=f"Nature MARL {self.version}"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Training command
        self._add_train_parser(subparsers)

        # Testing command
        self._add_test_parser(subparsers)

        # Benchmarking command
        self._add_benchmark_parser(subparsers)

        # Analysis command
        self._add_analyze_parser(subparsers)

        # Export command
        self._add_export_parser(subparsers)

        # Info command
        self._add_info_parser(subparsers)

        return parser

    def _add_train_parser(self, subparsers):
        """Add training command parser."""
        train_parser = subparsers.add_parser(
            "train",
            help="Train bio-inspired agents",
            description="Train multi-agent systems with bio-inspired communication"
        )

        train_parser.add_argument(
            "--config", "-c",
            type=str,
            help="Training configuration file (YAML/JSON)"
        )

        train_parser.add_argument(
            "--environment", "-e",
            type=str,
            default="simple_spread",
            choices=["simple_spread", "simple_tag", "waterworld", "multiwalker"],
            help="Environment to train on"
        )

        train_parser.add_argument(
            "--agents", "-a",
            type=int,
            default=4,
            help="Number of agents"
        )

        train_parser.add_argument(
            "--iterations", "-i",
            type=int,
            default=50,
            help="Number of training iterations"
        )

        train_parser.add_argument(
            "--output", "-o",
            type=str,
            default="./results",
            help="Output directory for results"
        )

        train_parser.add_argument(
            "--bio-preset",
            type=str,
            choices=["ant_colony", "bee_swarm", "neural_sync", "minimal"],
            help="Bio-inspired configuration preset"
        )

        train_parser.add_argument(
            "--wandb",
            action="store_true",
            help="Enable Weights & Biases logging"
        )

        train_parser.add_argument(
            "--gpu",
            action="store_true",
            help="Force GPU usage"
        )

        train_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with detailed logging"
        )

    def _add_test_parser(self, subparsers):
        """Add testing command parser."""
        test_parser = subparsers.add_parser(
            "test",
            help="Run system tests",
            description="Test installation, hardware, and functionality"
        )

        test_parser.add_argument(
            "--all",
            action="store_true",
            help="Run all tests"
        )

        test_parser.add_argument(
            "--quick",
            action="store_true",
            help="Run quick compatibility tests"
        )

        test_parser.add_argument(
            "--hardware",
            action="store_true",
            help="Test hardware compatibility (GPU, memory)"
        )

        test_parser.add_argument(
            "--modules",
            action="store_true",
            help="Test bio-inspired modules"
        )

        test_parser.add_argument(
            "--environments",
            action="store_true",
            help="Test environment compatibility"
        )

        test_parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output"
        )

    def _add_benchmark_parser(self, subparsers):
        """Add benchmarking command parser."""
        benchmark_parser = subparsers.add_parser(
            "benchmark",
            help="Run performance benchmarks",
            description="Benchmark bio-inspired system performance"
        )

        benchmark_parser.add_argument(
            "--agents",
            type=int,
            nargs="+",
            default=[4, 8, 16],
            help="Number of agents to benchmark"
        )

        benchmark_parser.add_argument(
            "--hidden-dim",
            type=int,
            nargs="+",
            default=[256, 512],
            help="Hidden dimensions to test"
        )

        benchmark_parser.add_argument(
            "--batch-size",
            type=int,
            nargs="+",
            default=[64, 128, 256],
            help="Batch sizes to test"
        )

        benchmark_parser.add_argument(
            "--iterations",
            type=int,
            default=100,
            help="Benchmark iterations"
        )

        benchmark_parser.add_argument(
            "--output", "-o",
            type=str,
            default="benchmark_results.json",
            help="Output file for benchmark results"
        )

        benchmark_parser.add_argument(
            "--gpu",
            action="store_true",
            help="Include GPU benchmarks"
        )

    def _add_analyze_parser(self, subparsers):
        """Add analysis command parser."""
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Analyze training results and bio-inspired metrics",
            description="Analyze bio-inspired behavior and performance"
        )

        analyze_parser.add_argument(
            "input_path",
            type=str,
            help="Path to results directory or metrics file"
        )

        analyze_parser.add_argument(
            "--metric",
            type=str,
            choices=["attention", "communication", "plasticity", "performance", "all"],
            default="all",
            help="Specific metric to analyze"
        )

        analyze_parser.add_argument(
            "--plot",
            action="store_true",
            help="Generate visualization plots"
        )

        analyze_parser.add_argument(
            "--export",
            type=str,
            help="Export analysis results to file"
        )

        analyze_parser.add_argument(
            "--compare",
            type=str,
            nargs="+",
            help="Compare multiple result directories"
        )

    def _add_export_parser(self, subparsers):
        """Add export command parser."""
        export_parser = subparsers.add_parser(
            "export",
            help="Export trained models",
            description="Export models in various formats"
        )

        export_parser.add_argument(
            "model_path",
            type=str,
            help="Path to trained model"
        )

        export_parser.add_argument(
            "--format",
            type=str,
            choices=["onnx", "torchscript", "tflite"],
            default="onnx",
            help="Export format"
        )

        export_parser.add_argument(
            "--output", "-o",
            type=str,
            help="Output file path"
        )

        export_parser.add_argument(
            "--optimize",
            action="store_true",
            help="Apply optimizations during export"
        )

    def _add_info_parser(self, subparsers):
        """Add info command parser."""
        info_parser = subparsers.add_parser(
            "info",
            help="Display system and package information",
            description="Show system configuration and package details"
        )

        info_parser.add_argument(
            "--system",
            action="store_true",
            help="Show system information"
        )

        info_parser.add_argument(
            "--models",
            action="store_true",
            help="Show available model configurations"
        )

        info_parser.add_argument(
            "--environments",
            action="store_true",
            help="Show available environments"
        )

    def run_train(self, args) -> bool:
        """Execute training command."""
        self.console.print(Panel.fit(
            "🚀 [bold green]Starting Bio-Inspired Training[/bold green]",
            border_style="green"
        ))

        # Load configuration
        config = self._load_training_config(args)

        # Setup logging
        experiment_name = f"bio_marl_{int(time.time())}"
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)

        with bio_inspired_logging_context(
            experiment_name=experiment_name,
            config=config,
            log_file=output_dir / "training.log"
        ) as exp_logger:

            # Initialize metrics tracking
            metrics_tracker = BioInspiredMetricsTracker(
                track_attention=True,
                track_communication=True,
                track_plasticity=True,
                track_performance=True
            )

            try:
                # Import training components
                import ray
                from ray.rllib.algorithms.ppo import PPOConfig

                # Initialize Ray
                ray.init(local_mode=args.debug)

                # Create environment
                env_fn = self._create_environment(args.environment, args.agents)

                # Create bio-inspired module spec
                module_spec = self._create_module_spec(args, config)

                # Configure PPO
                ppo_config = self._create_ppo_config(args, config, env_fn, module_spec)

                # Build algorithm
                algorithm = ppo_config.build()

                # Training loop with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:

                    task = progress.add_task(
                        f"Training {args.agents} agents...",
                        total=args.iterations
                    )

                    best_reward = float('-inf')
                    results = []

                    for iteration in range(args.iterations):
                        # Train one iteration
                        result = algorithm.train()
                        results.append(result)

                        # Update progress
                        reward = result['env_runners']['episode_reward_mean']
                        progress.update(
                            task,
                            advance=1,
                            description=f"Training (Reward: {reward:.2f})"
                        )

                        # Log bio-inspired metrics
                        self._log_training_metrics(result, metrics_tracker, exp_logger)

                        # Save best model
                        if reward > best_reward:
                            best_reward = reward
                            checkpoint_path = algorithm.save(str(output_dir / "checkpoints"))
                            exp_logger.info(f"New best model saved: {reward:.2f}")

                # Export final results
                self._export_training_results(results, metrics_tracker, output_dir)

                self.console.print(f"\n✅ [bold green]Training completed![/bold green]")
                self.console.print(f"📁 Results saved to: [cyan]{output_dir}[/cyan]")
                self.console.print(f"🏆 Best reward: [yellow]{best_reward:.2f}[/yellow]")

                # Cleanup
                ray.shutdown()
                return True

            except Exception as e:
                exp_logger.error(f"Training failed: {e}")
                self.console.print(f"❌ [bold red]Training failed:[/bold red] {e}")
                return False

    def run_test(self, args) -> bool:
        """Execute testing command."""
        self.console.print(Panel.fit(
            "🧪 [bold blue]Running System Tests[/bold blue]",
            border_style="blue"
        ))

        all_passed = True

        if args.all or args.quick:
            self.console.print("\n[bold]Quick Compatibility Tests[/bold]")
            passed = self._run_quick_tests(args.verbose)
            all_passed &= passed

        if args.all or args.hardware:
            self.console.print("\n[bold]Hardware Compatibility Tests[/bold]")
            passed = self._run_hardware_tests(args.verbose)
            all_passed &= passed

        if args.all or args.modules:
            self.console.print("\n[bold]Bio-Inspired Module Tests[/bold]")
            passed = self._run_module_tests(args.verbose)
            all_passed &= passed

        if args.all or args.environments:
            self.console.print("\n[bold]Environment Tests[/bold]")
            passed = self._run_environment_tests(args.verbose)
            all_passed &= passed

        # Summary
        if all_passed:
            self.console.print("\n✅ [bold green]All tests passed![/bold green]")
        else:
            self.console.print("\n❌ [bold red]Some tests failed.[/bold red]")

        return all_passed

    def run_benchmark(self, args) -> bool:
        """Execute benchmarking command."""
        self.console.print(Panel.fit(
            "⚡ [bold yellow]Performance Benchmarking[/bold yellow]",
            border_style="yellow"
        ))

        results = {}

        # Create benchmark table
        table = Table(title="Benchmark Results")
        table.add_column("Configuration", style="cyan")
        table.add_column("Forward Time (ms)", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Throughput (samples/s)", justify="right")

        total_configs = len(args.agents) * len(args.hidden_dim) * len(args.batch_size)

        with Progress(console=self.console) as progress:
            task = progress.add_task("Benchmarking...", total=total_configs)

            for num_agents in args.agents:
                for hidden_dim in args.hidden_dim:
                    for batch_size in args.batch_size:
                        config_name = f"{num_agents}a-{hidden_dim}h-{batch_size}b"

                        # Run benchmark
                        benchmark_result = self._run_single_benchmark(
                            num_agents, hidden_dim, batch_size, args.iterations, args.gpu
                        )

                        results[config_name] = benchmark_result

                        # Add to table
                        table.add_row(
                            config_name,
                            f"{benchmark_result['forward_time_ms']:.1f}",
                            f"{benchmark_result['memory_mb']:.1f}",
                            f"{benchmark_result['throughput']:.0f}"
                        )

                        progress.advance(task)

        self.console.print(table)

        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.console.print(f"\n📁 Benchmark results saved to: [cyan]{output_path}[/cyan]")
        return True

    def run_analyze(self, args) -> bool:
        """Execute analysis command."""
        self.console.print(Panel.fit(
            "📊 [bold magenta]Analyzing Bio-Inspired Metrics[/bold magenta]",
            border_style="magenta"
        ))

        try:
            # Load data
            data = self._load_analysis_data(args.input_path)

            # Perform analysis based on metric type
            if args.metric == "all":
                metrics = ["attention", "communication", "plasticity", "performance"]
            else:
                metrics = [args.metric]

            analysis_results = {}

            for metric in metrics:
                self.console.print(f"\n[bold]Analyzing {metric.title()} Metrics[/bold]")
                result = self._analyze_metric(data, metric, args.plot)
                analysis_results[metric] = result

                # Display summary
                self._display_metric_summary(metric, result)

            # Export results if requested
            if args.export:
                self._export_analysis_results(analysis_results, args.export)

            return True

        except Exception as e:
            self.console.print(f"❌ [bold red]Analysis failed:[/bold red] {e}")
            return False

    def run_info(self, args) -> bool:
        """Execute info command."""
        if args.system:
            self._display_system_info()

        if args.models:
            self._display_model_info()

        if args.environments:
            self._display_environment_info()

        if not any([args.system, args.models, args.environments]):
            # Show all info by default
            self._display_system_info()
            self._display_model_info()
            self._display_environment_info()

        return True

    def _display_system_info(self):
        """Display system information."""
        self.console.print(Panel.fit(
            "💻 [bold blue]System Information[/bold blue]",
            border_style="blue"
        ))

        # Create system info tree
        tree = Tree("🖥️ System Configuration")

        # Python info
        python_branch = tree.add("🐍 Python")
        python_branch.add(f"Version: {sys.version.split()[0]}")
        python_branch.add(f"Executable: {sys.executable}")

        # PyTorch info
        torch_branch = tree.add("🔥 PyTorch")
        torch_branch.add(f"Version: {torch.__version__}")
        torch_branch.add(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            torch_branch.add(f"CUDA Version: {torch.version.cuda}")
            torch_branch.add(f"GPU Count: {torch.cuda.device_count()}")

        # Hardware info
        try:
            import psutil
            hardware_branch = tree.add("⚡ Hardware")
            memory = psutil.virtual_memory()
            hardware_branch.add(f"CPU Cores: {psutil.cpu_count()}")
            hardware_branch.add(f"RAM: {memory.total / (1024**3):.1f} GB")
            hardware_branch.add(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        except ImportError:
            tree.add("⚠️ Hardware info unavailable (psutil not installed)")

        self.console.print(tree)

    def _display_model_info(self):
        """Display available model configurations."""
        self.console.print("\n[bold blue]📐 Available Model Configurations[/bold blue]")

        configs = {
            "Ant Colony": {
                "description": "Optimized for foraging and path-finding tasks",
                "agents": "8-12",
                "comm_rounds": "4",
                "use_positional_encoding": True,
                "plasticity_rate": 0.2
            },
            "Bee Swarm": {
                "description": "Spatial coordination with waggle dance communication",
                "agents": "6-8",
                "comm_rounds": "3",
                "use_positional_encoding": True,
                "plasticity_rate": 0.15
            },
            "Neural Sync": {
                "description": "Synchronized neural network dynamics",
                "agents": "4-6",
                "comm_rounds": "5",
                "use_positional_encoding": False,
                "plasticity_rate": 0.3
            },
            "Minimal": {
                "description": "Lightweight configuration for resource-constrained environments",
                "agents": "2-4",
                "comm_rounds": "2",
                "use_positional_encoding": False,
                "plasticity_rate": 0.1
            }
        }

        for name, config in configs.items():
            self.console.print(f"\n[cyan]{name}[/cyan]: {config['description']}")
            for key, value in config.items():
                if key != 'description':
                    self.console.print(f"  {key}: {value}")

    def _load_training_config(self, args) -> Dict[str, Any]:
        """Load training configuration from file or create from args."""
        if args.config:
            # Load from file
            config_path = Path(args.config)
            if config_path.suffix == '.json':
                with open(config_path) as f:
                    return json.load(f)
            else:
                # Assume YAML
                import yaml
                with open(config_path) as f:
                    return yaml.safe_load(f)
        else:
            # Create from command line arguments
            return self._create_config_from_args(args)

    def _create_config_from_args(self, args) -> Dict[str, Any]:
        """Create configuration from command line arguments."""
        config = {
            "num_agents": args.agents,
            "environment": args.environment,
            "iterations": args.iterations,
            "use_communication": True,
            "debug_mode": args.debug
        }

        # Apply bio-inspired presets
        if args.bio_preset:
            preset_configs = {
                "ant_colony": {
                    "hidden_dim": 512,
                    "memory_dim": 128,
                    "comm_channels": 24,
                    "comm_rounds": 4,
                    "plasticity_rate": 0.2,
                    "use_positional_encoding": True,
                    "adaptive_plasticity": True
                },
                "bee_swarm": {
                    "hidden_dim": 256,
                    "memory_dim": 64,
                    "comm_channels": 16,
                    "comm_rounds": 3,
                    "plasticity_rate": 0.15,
                    "use_positional_encoding": True,
                    "adaptive_plasticity": True
                },
                "neural_sync": {
                    "hidden_dim": 384,
                    "memory_dim": 96,
                    "comm_channels": 32,
                    "comm_rounds": 5,
                    "plasticity_rate": 0.3,
                    "use_positional_encoding": False,
                    "adaptive_plasticity": True
                },
                "minimal": {
                    "hidden_dim": 128,
                    "memory_dim": 32,
                    "comm_channels": 8,
                    "comm_rounds": 2,
                    "plasticity_rate": 0.1,
                    "use_positional_encoding": False,
                    "adaptive_plasticity": False
                }
            }
            config.update(preset_configs[args.bio_preset])

        return config

    def main(self):
        """Main CLI entry point."""
        parser = self.create_parser()
        args = parser.parse_args()

        if not args.command:
            # Show help if no command provided
            parser.print_help()
            return 1

        # Display header
        self.console.print(Panel.fit(
            f"🌿 [bold green]Nature MARL CLI v{self.version}[/bold green]\n"
            "Bio-Inspired Multi-Agent Reinforcement Learning",
            border_style="green"
        ))

        # Execute command
        success = False
        try:
            if args.command == "train":
                success = self.run_train(args)
            elif args.command == "test":
                success = self.run_test(args)
            elif args.command == "benchmark":
                success = self.run_benchmark(args)
            elif args.command == "analyze":
                success = self.run_analyze(args)
            elif args.command == "export":
                success = self.run_export(args)
            elif args.command == "info":
                success = self.run_info(args)
            else:
                self.console.print(f"❌ Unknown command: {args.command}")
                return 1

        except KeyboardInterrupt:
            self.console.print("\n⚠️  [yellow]Operation cancelled by user[/yellow]")
            return 1
        except Exception as e:
            self.console.print(f"❌ [bold red]Unexpected error:[/bold red] {e}")
            if args.debug if hasattr(args, 'debug') else False:
                import traceback
                traceback.print_exc()
            return 1

        return 0 if success else 1


def main():
    """CLI entry point."""
    cli = NatureMARL_CLI()
    sys.exit(cli.main())


if __name__ == "__main__":
    main()


# File: nature_marl/__init__.py
"""
Nature-Inspired Multi-Agent Reinforcement Learning

A production-ready system implementing biological communication patterns
for intelligent multi-agent coordination.
"""

__version__ = "1.0.0"
__author__ = "Nature MARL Team"
__email__ = "team@nature-marl.org"

# Core exports
from nature_marl.core.production_bio_inspired_rl_module import (
    ProductionUnifiedBioInspiredRLModule,
    ProductionPheromoneAttentionNetwork,
    ProductionNeuralPlasticityMemory,
    ProductionMultiActionHead,
    create_production_bio_module_spec,
)

# Utility exports
from nature_marl.utils.logging_config import (
    BioInspiredMetricsTracker,
    BioInspiredVisualizer,
    setup_production_logging,
    bio_inspired_logging_context,
    performance_monitor,
)

# Training utilities
from nature_marl.training.config_factory import ConfigFactory
from nature_marl.training.callbacks import BioInspiredCallbacks

# Environment utilities
from nature_marl.environments.wrappers import BioInspiredWrapper

__all__ = [
    # Core module
    "ProductionUnifiedBioInspiredRLModule",
    "ProductionPheromoneAttentionNetwork",
    "ProductionNeuralPlasticityMemory",
    "ProductionMultiActionHead",
    "create_production_bio_module_spec",

    # Logging and monitoring
    "BioInspiredMetricsTracker",
    "BioInspiredVisualizer",
    "setup_production_logging",
    "bio_inspired_logging_context",
    "performance_monitor",

    # Training
    "ConfigFactory",
    "BioInspiredCallbacks",

    # Environments
    "BioInspiredWrapper",

    # Metadata
    "__version__",
    "__author__",
    "__email__",
]


# File: setup.py
"""
Setup script for Nature MARL package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "gymnasium>=0.28.1",
    "ray[rllib]>=2.5.0",
    "numpy>=1.24.0",
    "pettingzoo>=1.24.0",
    "supersuit>=3.8.0",
    "rich>=13.0.0",  # For CLI
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-benchmark>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "bandit>=1.7.0",
        "safety>=2.0.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinx-autodoc-typehints>=1.20.0",
        "myst-parser>=2.0.0",
    ],
    "viz": [
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "wandb>=0.15.0",
    ],
    "gpu": [
        "psutil>=5.9.0",
    ],
}

# All optional dependencies
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="nature-marl",
    version="1.0.0",
    description="Production-ready bio-inspired multi-agent reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nature MARL Team",
    author_email="team@nature-marl.org",
    url="https://github.com/your-org/nature-marl",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "nature-marl=nature_marl.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "reinforcement-learning",
        "multi-agent",
        "bio-inspired",
        "swarm-intelligence",
        "neural-plasticity",
        "pheromone-communication",
        "rllib",
        "pytorch",
    ],
    project_urls={
        "Documentation": "https://nature-marl.readthedocs.io/",
        "Source": "https://github.com/your-org/nature-marl",
        "Tracker": "https://github.com/your-org/nature-marl/issues",
    },
    include_package_data=True,
    zip_safe=False,
)
