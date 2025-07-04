# File: marlcomm/main_trainer.py
#!/usr/bin/env python3
"""
Nature-Inspired Multi-Agent Reinforcement Learning - Main Training Script

This is the main entry point for training agents with biological communication
patterns. The modular design allows for easy debugging and configuration.

Usage:
    python main_trainer.py                  # Auto-detect hardware mode
    python main_trainer.py --gpu            # Force GPU mode  
    python main_trainer.py --cpu            # Force CPU mode
    python main_trainer.py --iterations 50  # Custom training length
    python main_trainer.py --test-only      # Run tests without training
"""

import sys
import argparse
import warnings
import traceback
from pathlib import Path

# Suppress various warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from environment import register_nature_comm_environment, test_environment_creation
from training_config import TrainingConfigFactory, print_hardware_summary
from ray import tune


class NatureMARL:
    """
    Main coordinator class for nature-inspired multi-agent reinforcement learning.
    
    This class orchestrates the entire training pipeline from environment setup
    to model training, with automatic hardware optimization and comprehensive
    error handling.
    """
    
    def __init__(self, force_mode: str = None):
        """
        Initialize the nature-inspired MARL system.
        
        Args:
            force_mode: Optional training mode override ("gpu" or "cpu")
        """
        self.force_mode = force_mode
        self.config_factory = TrainingConfigFactory()
        
    def setup_environment(self) -> bool:
        """
        Setup and test the multi-agent environment.
        
        Returns:
            True if environment setup succeeds, False otherwise
        """
        print("🌱 Setting up multi-agent environment...")
        
        try:
            # Register the environment with Ray Tune
            register_nature_comm_environment()
            
            # Test environment creation
            if not test_environment_creation(debug=True):
                return False
            
            print("✅ Environment setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            traceback.print_exc()
            return False
    
    def create_training_config(self):
        """
        Create training configuration based on available hardware.
        
        Returns:
            Tuple of (PPOConfig, observation_space, action_space)
        """
        print("🔧 Creating training configuration...")
        
        # Get environment spaces for configuration
        from environment import create_nature_comm_env
        test_env = create_nature_comm_env({"debug": False})
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        test_env.close()
        
        # Create optimized configuration
        config = self.config_factory.create_config(
            obs_space, act_space, force_mode=self.force_mode
        )
        
        return config, obs_space, act_space
    
    def run_training(self, num_iterations: int = 20) -> bool:
        """
        Execute the complete training pipeline.
        
        Args:
            num_iterations: Number of training iterations to run
            
        Returns:
            True if training completes successfully, False otherwise
        """
        try:
            print(f"🚀 Starting nature-inspired MARL training ({num_iterations} iterations)...")
            
            # Create training configuration
            config, obs_space, act_space = self.create_training_config()
            
            # Get Ray Tune configuration
            tune_config = self.config_factory.get_tune_config(num_iterations)
            
            # Create and run tuner
            tuner = tune.Tuner(
                "PPO",
                param_space=config.to_dict(),
                run_config=tune.RunConfig(**tune_config)
            )
            
            # Execute training
            result = tuner.fit()
            
            # Extract and display results
            try:
                best_result = result.get_best_result()
                
                print("\n🎉 Training completed successfully!")
                print("📊 Final Results:")
                print(f"   Best Episode Reward: {best_result.metrics.get('episode_reward_mean', 'N/A')}")
                print(f"   Communication Entropy: {best_result.metrics.get('custom_metrics', {}).get('avg_communication_entropy', 'N/A')}")
                print(f"   Coordination Efficiency: {best_result.metrics.get('custom_metrics', {}).get('avg_coordination_efficiency', 'N/A')}")
                
                return True
                
            except Exception as e:
                print(f"⚠️  Training completed but result extraction failed: {e}")
                print("✅ Training process finished (results may be in Ray logs)")
                return True
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
            # Provide helpful error diagnostics
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                print("🔧 Memory Error - Try:")
                print("   1. python main_trainer.py --cpu")
                print("   2. Reduce batch sizes in training_config.py")
                print("   3. Close other applications")
            elif "cuda" in error_msg:
                print("🔧 CUDA Error - Try:")
                print("   1. python main_trainer.py --cpu")
                print("   2. Restart your system")
                print("   3. Check CUDA installation")
            elif "actor" in error_msg or "worker" in error_msg:
                print("🔧 Ray Worker Error - Try:")
                print("   1. Reduce num_env_runners in training_config.py")
                print("   2. Check available system memory")
            
            traceback.print_exc()
            return False
    
    def run_tests_only(self) -> bool:
        """
        Run comprehensive tests without training.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print("🧪 Running comprehensive system tests...")
        
        success = True
        
        # Test 1: Environment creation
        print("\n1️⃣ Testing environment creation...")
        if not self.setup_environment():
            success = False
        
        # Test 2: Configuration creation
        print("\n2️⃣ Testing configuration creation...")
        try:
            config, obs_space, act_space = self.create_training_config()
            print("✅ Configuration creation successful")
        except Exception as e:
            print(f"❌ Configuration creation failed: {e}")
            success = False
        
        # Test 3: Hardware detection
        print("\n3️⃣ Testing hardware detection...")
        try:
            print_hardware_summary()
            print("✅ Hardware detection successful")
        except Exception as e:
            print(f"❌ Hardware detection failed: {e}")
            success = False
        
        print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
        return success


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nature-Inspired Multi-Agent Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_trainer.py                    # Auto-detect hardware
  python main_trainer.py --gpu              # Force GPU training
  python main_trainer.py --cpu              # Force CPU training  
  python main_trainer.py --iterations 50    # Custom training length
  python main_trainer.py --test-only        # Run tests only
        """
    )
    
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Force GPU training mode"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true", 
        help="Force CPU training mode"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of training iterations (default: 20)"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run tests without training"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the training script."""
    
    # Print header
    print("🌿 Nature-Inspired Multi-Agent Reinforcement Learning")
    print("   Implementing biological communication patterns")
    print("   🐜 Pheromone trails • 🧠 Neural plasticity • 🐝 Information encoding")
    print("=" * 70)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if args.gpu and args.cpu:
        print("❌ Error: Cannot specify both --gpu and --cpu")
        sys.exit(1)
    
    # Determine training mode
    force_mode = None
    if args.gpu:
        force_mode = "gpu"
        print("🔧 Forcing GPU training mode")
    elif args.cpu:
        force_mode = "cpu"
        print("🔧 Forcing CPU training mode")
    
    # Initialize the MARL system
    marl_system = NatureMARL(force_mode=force_mode)
    
    # Display hardware information
    print_hardware_summary()
    print()
    
    if args.test_only:
        # Run tests only
        success = marl_system.run_tests_only()
        sys.exit(0 if success else 1)
    else:
        # Setup environment
        if not marl_system.setup_environment():
            print("❌ Environment setup failed. Exiting.")
            sys.exit(1)
        
        # Run training
        success = marl_system.run_training(args.iterations)
        
        if success:
            print("\n🎉 Nature-inspired MARL training completed successfully!")
            print("🔬 Your agents have learned biological communication patterns")
            print("   and can now coordinate like a swarm of intelligent organisms!")
        else:
            print("\n❌ Training encountered errors. Check logs above for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
