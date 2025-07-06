#!/usr/bin/env python3
"""
Main trainer for bio-inspired multi-agent reinforcement learning.
Updated for RLlib 2.9.x with the new API stack.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from marlcomm package
from marlcomm.config.training_config import (
    create_config_for_experiment,
    list_experiments,
    EXPERIMENT_CONFIGS
)
from marlcomm.utils.callbacks import BioInspiredCallbacks
from marlcomm.utils.logging_config import setup_logging
from marlcomm.environments.emergence_environments import (
    CommunicationEnv,  # Adjust based on your actual environment
)

# Import torch to check CUDA
torch, _ = try_import_torch()


def register_environments():
    """Register all custom environments with Ray."""
    # Register your environments here
    register_env("CommEnv-v0", lambda config: CommunicationEnv(config))
    # Add more environments as needed
    # register_env("ForagingEnv-v0", lambda config: ForagingEnv(config))
    # register_env("NavigationEnv-v0", lambda config: NavigationEnv(config))


def train(args: argparse.Namespace):
    """Main training function."""

    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )

    # Initialize Ray
    ray_init_config = {
        "local_mode": args.debug,
        "num_cpus": args.ray_num_cpus,
        "num_gpus": args.ray_num_gpus,
    }

    if args.ray_address:
        ray_init_config["address"] = args.ray_address

    logger.info(f"Initializing Ray with config: {ray_init_config}")
    ray.init(**ray_init_config)

    # Register environments
    register_environments()

    # Create training configuration
    logger.info(f"Loading experiment configuration: {args.experiment}")

    # Override experiment config with CLI arguments
    overrides = {}
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.train_batch_size is not None:
        overrides["train_batch_size"] = args.train_batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.num_gpus is not None:
        overrides["num_gpus"] = args.num_gpus
    if args.seed is not None:
        overrides["seed"] = args.seed

    # Add callbacks
    overrides["callbacks_class"] = BioInspiredCallbacks

    # Create config
    config = create_config_for_experiment(args.experiment, **overrides)

    # Set up logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / args.experiment / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = config.to_dict()
    with open(log_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Logging to: {log_dir}")

    if args.tune:
        # Hyperparameter tuning with Ray Tune
        logger.info("Starting hyperparameter search with Ray Tune")

        # Define search space
        tune_config = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.0, 0.1),
            "clip_param": tune.uniform(0.1, 0.3),
            "train_batch_size": tune.choice([4000, 8000, 16000]),
        }

        # Add bio-inspired hyperparameters
        if EXPERIMENT_CONFIGS[args.experiment].get("use_plasticity", False):
            tune_config["plasticity_rate"] = tune.loguniform(1e-4, 1e-2)

        if EXPERIMENT_CONFIGS[args.experiment].get("use_pheromones", False):
            tune_config["pheromone_decay_rate"] = tune.uniform(0.9, 0.99)

        # Run tuning
        analysis = tune.run(
            PPO,
            config=config.to_dict(),
            param_space=tune_config,
            num_samples=args.num_samples,
            stop={
                "timesteps_total": EXPERIMENT_CONFIGS[args.experiment]["total_timesteps"],
                "episode_reward_mean": args.target_reward,
            },
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=str(log_dir),
            verbose=1 if not args.quiet else 0,
            progress_reporter=tune.CLIReporter(
                metric_columns=["episode_reward_mean", "timesteps_total"],
                parameter_columns=list(tune_config.keys()),
            ),
        )

        # Print results
        best_trial = analysis.get_best_trial("episode_reward_mean", "max")
        logger.info(f"\nBest trial config: {best_trial.config}")
        logger.info(f"Best trial final reward: {best_trial.last_result['episode_reward_mean']}")

        # Save best config
        with open(log_dir / "best_config.json", "w") as f:
            json.dump(best_trial.config, f, indent=2)

    else:
        # Single training run
        logger.info("Starting single training run")

        # Build algorithm
        algo = config.build()

        # Training loop
        iteration = 0
        try:
            while True:
                # Train for one iteration
                result = algo.train()
                iteration += 1

                # Log progress
                if iteration % args.print_freq == 0:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Iteration: {iteration}")
                    logger.info(f"Timesteps: {result['timesteps_total']:,}")
                    logger.info(f"Episodes: {result['episodes_total']:,}")
                    logger.info(f"Reward mean: {result['episode_reward_mean']:.3f}")
                    logger.info(f"Reward min/max: {result['episode_reward_min']:.3f} / {result['episode_reward_max']:.3f}")

                    # Log custom metrics
                    if "custom_metrics" in result:
                        logger.info("\nCustom Metrics:")
                        for key, value in result["custom_metrics"].items():
                            if isinstance(value, (int, float)):
                                logger.info(f"  {key}: {value:.4f}")

                    # Log bio-inspired metrics if available
                    if "bio_inspired" in result:
                        logger.info("\nBio-Inspired Metrics:")
                        for key, value in result["bio_inspired"].items():
                            logger.info(f"  {key}: {value:.4f}")

                # Save checkpoint
                if iteration % args.checkpoint_freq == 0:
                    checkpoint_dir = algo.save(str(log_dir / "checkpoints"))
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")

                # Check stopping conditions
                if result["timesteps_total"] >= EXPERIMENT_CONFIGS[args.experiment]["total_timesteps"]:
                    logger.info(f"\nReached target timesteps. Training complete!")
                    break

                if args.target_reward and result["episode_reward_mean"] >= args.target_reward:
                    logger.info(f"\nReached target reward. Training complete!")
                    break

                if args.max_iterations and iteration >= args.max_iterations:
                    logger.info(f"\nReached max iterations. Training complete!")
                    break

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")

        finally:
            # Save final checkpoint
            final_checkpoint = algo.save(str(log_dir / "final_model"))
            logger.info(f"Final model saved: {final_checkpoint}")

            # Save training history
            with open(log_dir / "training_history.json", "w") as f:
                json.dump({"iterations": iteration}, f)

            # Cleanup
            algo.stop()

    ray.shutdown()
    logger.info("Training completed successfully!")


def evaluate(args: argparse.Namespace):
    """Evaluate a trained model."""

    logger = setup_logging(log_level=args.log_level)

    # Initialize Ray
    ray.init(local_mode=True)
    register_environments()

    # Load configuration from checkpoint directory
    checkpoint_path = Path(args.checkpoint_path)
    config_path = checkpoint_path.parent.parent / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning("No saved config found, using default experiment config")
        saved_config = create_config_for_experiment(args.experiment).to_dict()

    # Create config for evaluation
    config = PPO.from_dict(saved_config)
    config.rollouts(num_rollout_workers=0)  # No workers for evaluation
    config.explore = False  # Disable exploration

    # Build and restore
    algo = config.build()
    algo.restore(str(checkpoint_path))

    # Create environment
    env_config = saved_config.get("env_config", {})
    env = CommunicationEnv(env_config)  # Adjust based on your environment

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    custom_metrics = {}

    for episode in range(args.num_eval_episodes):
        obs, info = env.reset()
        done = {"__all__": False}
        episode_reward = 0
        episode_length = 0

        while not done["__all__"]:
            actions = {}

            # Get actions for all agents
            for agent_id, agent_obs in obs.items():
                policy_id = algo.config.policy_mapping_fn(agent_id, None, None)
                action = algo.compute_single_action(
                    agent_obs,
                    policy_id=policy_id,
                    explore=False
                )
                actions[agent_id] = action

            # Step environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = {"__all__": terminateds.get("__all__", False) or truncateds.get("__all__", False)}

            episode_reward += sum(rewards.values())
            episode_length += 1

            # Collect custom metrics
            for agent_id, info in infos.items():
                for key, value in info.items():
                    if key not in custom_metrics:
                        custom_metrics[key] = []
                    custom_metrics[key].append(value)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            logger.info(f"Evaluated {episode + 1}/{args.num_eval_episodes} episodes")

    # Print evaluation results
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation Results:")
    logger.info(f"Episodes: {args.num_eval_episodes}")
    logger.info(f"Average Reward: {sum(episode_rewards) / len(episode_rewards):.3f}")
    logger.info(f"Std Reward: {torch.tensor(episode_rewards).std():.3f}")
    logger.info(f"Min/Max Reward: {min(episode_rewards):.3f} / {max(episode_rewards):.3f}")
    logger.info(f"Average Length: {sum(episode_lengths) / len(episode_lengths):.1f}")

    # Print custom metrics
    if custom_metrics:
        logger.info("\nCustom Metrics:")
        for key, values in custom_metrics.items():
            if all(isinstance(v, (int, float)) for v in values):
                avg_value = sum(values) / len(values)
                logger.info(f"  {key}: {avg_value:.4f}")

    ray.shutdown()


def analyze(args: argparse.Namespace):
    """Analyze training results and create visualizations."""
    from analysis.analysis_visualization import create_training_plots

    logger = setup_logging(log_level=args.log_level)

    # Find all experiments in log directory
    log_dir = Path(args.log_dir)
    experiments = [d for d in log_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(experiments)} experiments in {log_dir}")

    # Create analysis for each experiment
    for exp_dir in experiments:
        logger.info(f"\nAnalyzing: {exp_dir.name}")

        # Find all runs for this experiment
        runs = [d for d in exp_dir.iterdir() if d.is_dir()]

        for run_dir in runs:
            # Check if this is a training run
            progress_file = run_dir / "progress.csv"
            if progress_file.exists():
                logger.info(f"  Creating plots for: {run_dir.name}")

                # Create visualization
                output_dir = run_dir / "analysis"
                output_dir.mkdir(exist_ok=True)

                create_training_plots(
                    progress_file,
                    output_dir,
                    experiment_name=exp_dir.name
                )

    logger.info("\nAnalysis complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bio-inspired Multi-Agent RL Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log to file (in addition to console)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List experiments command
    list_parser = subparsers.add_parser("list", help="List available experiments")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "experiment",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment configuration to use"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of workers"
    )
    train_parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override number of GPUs"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    train_parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override training batch size"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    train_parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Checkpoint frequency (iterations)"
    )
    train_parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Print frequency (iterations)"
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        default="./results",
        help="Directory for logs and checkpoints"
    )
    train_parser.add_argument(
        "--tune",
        action="store_true",
        help="Use Ray Tune for hyperparameter search"
    )
    train_parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of trials for Ray Tune"
    )
    train_parser.add_argument(
        "--target-reward",
        type=float,
        default=None,
        help="Stop training when this reward is reached"
    )
    train_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum training iterations"
    )
    train_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (local Ray)"
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    train_parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address"
    )
    train_parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Number of CPUs for Ray"
    )
    train_parser.add_argument(
        "--ray-num-gpus",
        type=int,
        default=None,
        help="Number of GPUs for Ray"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help="Experiment configuration (if config.json not found)"
    )
    eval_parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training results")
    analyze_parser.add_argument(
        "--log-dir",
        type=str,
        default="./results",
        help="Directory containing training logs"
    )

    args = parser.parse_args()

    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Execute command
    if args.command == "list":
        list_experiments()
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
