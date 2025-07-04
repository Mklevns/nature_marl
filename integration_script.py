# File: marlcomm/run_research_experiment.py
"""
Complete Integration Script for Nature-Inspired MARL Communication Research

This script demonstrates how all modules work together to:
1. Define research hypotheses
2. Create appropriate environments and rewards
3. Train agents with emergent communication
4. Analyze communication patterns
5. Generate comprehensive reports

Usage:
    python run_research_experiment.py --experiment ant_foraging
    python run_research_experiment.py --experiment all --num_seeds 5
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Import all our custom modules
from research_framework import (
    ResearchHypothesis, ExperimentConfig, ResearchExperiment,
    HypothesisGenerator, CommunicationParadigm, EmergencePressure
)
from emergence_environments import (
    create_emergence_environment, ForagingEnvironment,
    PredatorPreyEnvironment, TemporalCoordinationEnvironment
)
from communication_metrics import (
    CommunicationAnalyzer, CommunicationEvent, EmergenceDetector
)
from reward_engineering import (
    RewardEngineer, RewardContext, RewardPresets
)
from analysis_visualization import (
    ExperimentData, ExperimentReporter, CommunicationVisualizer,
    EmergenceAnalyzer, InteractiveDashboard
)
from rl_module import NatureInspiredCommModule, create_nature_comm_module_spec
from training_config import TrainingConfigFactory


class IntegratedExperiment(ResearchExperiment):
    """
    Complete implementation of a research experiment integrating all modules.
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: Path):
        super().__init__(config, output_dir)
        
        # Initialize components
        self.communication_analyzer = CommunicationAnalyzer()
        self.reward_engineer = None
        self.env = None
        self.algo = None
        self.collected_events = []
        self.metrics_history = {
            'mutual_information': [],
            'coordination_efficiency': [],
            'protocol_stability': [],
            'network_density': [],
            'channel_capacity': [],
            'semantic_coherence': []
        }
        self.episode_rewards = []
        self.agent_trajectories = {}
        
    def setup_environment(self):
        """Create emergence environment based on hypothesis."""
        print(f"🌍 Setting up {self.config.environment_type} environment...")
        
        # Map environment type to creation parameters
        env_params = {
            'n_agents': self.config.num_agents,
            'grid_size': (30, 30),
            'episode_length': 200
        }
        
        # Add specific parameters based on environment
        if self.config.environment_type == "ForagingEnvironment":
            env_params.update({
                'n_food_clusters': 3,
                'cluster_size': 5
            })
            env_type = "foraging"
        elif self.config.environment_type == "PredatorPreyEnvironment":
            env_params.update({
                'n_predators': 2,
                'predator_speed': 1.2
            })
            env_type = "predator_prey"
        elif self.config.environment_type == "TemporalCoordinationEnvironment":
            env_params.update({
                'n_switches': 3,
                'sync_window': 5
            })
            env_type = "temporal_coordination"
        else:
            env_type = "foraging"  # Default
            
        # Create environment
        self.env = create_emergence_environment(env_type, **env_params)
        
        # Wrap for RLlib
        self.env = ParallelPettingZooEnv(self.env)
        
        print(f"✅ Environment created: {env_type}")
        
    def setup_agents(self):
        """Initialize agents with nature-inspired communication modules."""
        print("🤖 Setting up agents with communication modules...")
        
        # Get observation and action spaces
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        
        # Setup reward engineering based on hypothesis
        self._setup_rewards()
        
        # Create training configuration
        config_factory = TrainingConfigFactory()
        
        # Create PPO configuration with our custom RL module
        ppo_config = (
            PPOConfig()
            .environment(
                env=lambda config: self._create_env_with_rewards(config),
                env_config={"experiment": self}
            )
            .framework("torch")
            .env_runners(
                num_env_runners=4,
                num_envs_per_env_runner=1
            )
            .rl_module(
                rl_module_spec=create_nature_comm_module_spec(
                    obs_space,
                    act_space,
                    model_config=self.config.communication_channels
                )
            )
            .training(
                train_batch_size_per_learner=1024,
                minibatch_size=128,
                lr=3e-4,
                num_epochs=10
            )
            .multi_agent(
                policies={
                    "shared_policy": None  # All agents share the same policy
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
            )
            .callbacks(CommunicationTrackingCallbacks)
        )
        
        # Build algorithm
        self.algo = ppo_config.build()
        
        print("✅ Agents initialized with nature-inspired communication")
        
    def _setup_rewards(self):
        """Setup reward engineering based on hypothesis."""
        hypothesis_id = self.config.hypothesis_id
        
        # Select appropriate reward preset based on hypothesis
        if "ant" in hypothesis_id:
            preset = RewardPresets.ant_colony_foraging()
        elif "bee" in hypothesis_id:
            preset = RewardPresets.bee_waggle_dance()
        elif "neural" in hypothesis_id:
            preset = RewardPresets.neural_synchronization()
        else:
            # Default multi-principle reward
            self.reward_engineer = RewardEngineer(self.config.environment_type)
            self.reward_engineer.create_implicit_reward_structure("multi_principle")
            return
            
        self.reward_engineer = preset['engineer']
        print(f"✅ Reward engineering: {preset['description']}")
        
    def _create_env_with_rewards(self, env_config):
        """Create environment with integrated reward engineering."""
        # Create base environment
        env = self.setup_environment()
        
        # Wrap with reward engineering
        return RewardEngineeringWrapper(env, self.reward_engineer, self)
        
    def run_training_step(self):
        """Execute one training iteration and collect metrics."""
        # Train
        result = self.algo.train()
        
        # Extract metrics from result
        step_metrics = self._extract_metrics_from_result(result)
        
        # Update emergence detector
        emergence_status = self.communication_analyzer.emergence_detector.update(
            self.collected_events[-100:]  # Last 100 events
        )
        
        # Record metrics
        for metric_name, value in emergence_status['current_metrics'].items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)
                
        # Check for emergence
        if emergence_status['emergence_detected'] and self.results['emergence_timestep'] is None:
            self.results['emergence_timestep'] = len(self.episode_rewards)
            print(f"🎉 Communication emerged at episode {self.results['emergence_timestep']}!")
            
        # Record episode rewards
        if 'env_runners' in result:
            episode_reward = result['env_runners'].get('episode_reward_mean', 0)
            self.episode_rewards.append(episode_reward)
            
        return step_metrics
        
    def _extract_metrics_from_result(self, result):
        """Extract emergence metrics from training result."""
        from communication_metrics import EmergenceMetrics
        
        # Create metrics object
        metrics = EmergenceMetrics()
        
        # Extract from custom metrics if available
        if 'env_runners' in result and 'custom_metrics' in result['env_runners']:
            custom = result['env_runners']['custom_metrics']
            
            metrics.mutual_information = custom.get('mutual_information_mean', 0.0)
            metrics.coordination_efficiency = custom.get('coordination_efficiency_mean', 0.0)
            metrics.communication_frequency = custom.get('communication_frequency_mean', 0.0)
            
        return metrics
        
    def analyze_communication_patterns(self):
        """Analyze emerged communication patterns."""
        print("🔍 Analyzing communication patterns...")
        
        # Full analysis
        analysis_results = self.communication_analyzer.analyze_episode(self.collected_events)
        
        # Extract key findings
        findings = {
            'protocols_detected': len(analysis_results['detected_protocols']),
            'semantic_clusters': analysis_results['semantic_analysis']['message_clusters'],
            'emergence_score': analysis_results['emergence_status']['emergence_score'],
            'key_insights': analysis_results['insights']
        }
        
        print(f"✅ Analysis complete: {findings['protocols_detected']} protocols detected")
        
        return findings


class RewardEngineeringWrapper:
    """Wrapper to integrate reward engineering with environment."""
    
    def __init__(self, env, reward_engineer, experiment):
        self.env = env
        self.reward_engineer = reward_engineer
        self.experiment = experiment
        self._agent_list = list(env.agents)
        
    def __getattr__(self, name):
        """Delegate to wrapped environment."""
        return getattr(self.env, name)
        
    def step(self, actions):
        """Step with reward engineering."""
        # Original step
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        
        # Create reward context
        context = RewardContext(
            agent_states={agent: obs[agent] for agent in actions},
            agent_actions=actions,
            agent_observations=obs,
            communication_signals=self._extract_communication_signals(obs),
            environmental_state=self._get_env_state(),
            timestep=self.env.current_step if hasattr(self.env, 'current_step') else 0
        )
        
        # Apply reward engineering
        engineered_rewards = {}
        for agent in actions:
            if agent in rewards:  # Only for active agents
                total_reward, components = self.reward_engineer.calculate_reward(agent, context)
                engineered_rewards[agent] = rewards[agent] + total_reward
                
                # Store component breakdown in info
                if agent not in infos:
                    infos[agent] = {}
                infos[agent]['reward_components'] = components
                
        # Collect communication events
        self._collect_communication_events(obs, actions, context)
        
        return obs, engineered_rewards, terminateds, truncateds, infos
        
    def _extract_communication_signals(self, observations):
        """Extract communication signals from observations."""
        signals = {}
        
        for agent, obs in observations.items():
            # Assuming last N dimensions are communication
            # Adjust based on your observation space
            if len(obs) > 10:
                signals[agent] = obs[-8:]  # Last 8 dims for communication
                
        return signals
        
    def _get_env_state(self):
        """Get current environment state."""
        state = {}
        
        if hasattr(self.env, 'env'):  # PettingZoo wrapper
            actual_env = self.env.env
            
            # Extract relevant state based on environment type
            if hasattr(actual_env, 'collected_resources'):
                state['collective_score'] = actual_env.collected_resources
            if hasattr(actual_env, 'resources'):
                state['total_resources'] = len(actual_env.resources)
            if hasattr(actual_env, 'predator_positions'):
                state['n_predators'] = len(actual_env.predator_positions)
                
        return state
        
    def _collect_communication_events(self, obs, actions, context):
        """Collect communication events for analysis."""
        # Create communication events
        for sender in actions:
            if sender in context.communication_signals:
                # Determine receivers (all other agents in range)
                receivers = [agent for agent in actions if agent != sender]
                
                event = CommunicationEvent(
                    timestep=context.timestep,
                    sender_id=sender,
                    receiver_ids=receivers,
                    message=context.communication_signals[sender],
                    sender_state=obs[sender],
                    environmental_context=context.environmental_state,
                    response_actions={r: actions[r] for r in receivers if r in actions}
                )
                
                self.experiment.collected_events.append(event)


class CommunicationTrackingCallbacks:
    """RLlib callbacks for tracking communication metrics."""
    
    def on_episode_start(self, *, episode, **kwargs):
        """Initialize episode metrics."""
        episode.custom_metrics["communication_frequency"] = 0
        episode.custom_metrics["coordination_success"] = 0
        
    def on_episode_step(self, *, episode, **kwargs):
        """Track communication during episode."""
        # Count communication events
        for agent in episode.get_agents():
            agent_info = episode.last_info_for(agent)
            if agent_info and 'communication_sent' in agent_info:
                episode.custom_metrics["communication_frequency"] += 1
                
    def on_episode_end(self, *, episode, **kwargs):
        """Calculate final episode metrics."""
        # Normalize by episode length
        episode.custom_metrics["communication_frequency"] /= episode.length


class ResearchCampaignRunner:
    """Orchestrate complete research campaigns."""
    
    def __init__(self, output_base: Path = Path("research_output")):
        self.output_base = output_base
        self.output_base.mkdir(exist_ok=True)
        self.campaign_dir = None
        
    def run_hypothesis_test(self,
                          hypothesis: ResearchHypothesis,
                          num_seeds: int = 3,
                          training_steps: int = 10000) -> List[ExperimentData]:
        """Test a single hypothesis with multiple random seeds."""
        print(f"\n🧬 Testing Hypothesis: {hypothesis.id}")
        print(f"   {hypothesis.description}")
        
        experiment_results = []
        
        for seed in range(num_seeds):
            print(f"\n   Seed {seed + 1}/{num_seeds}")
            
            # Create experiment configuration
            config = ExperimentConfig(
                hypothesis_id=hypothesis.id,
                environment_type=self._get_environment_for_pressure(hypothesis.pressure),
                agent_architecture="NatureInspiredCommModule",
                communication_channels=self._get_channels_for_paradigm(hypothesis.paradigm),
                reward_structure=hypothesis.parameters,
                training_steps=training_steps,
                num_agents=5,
                num_seeds=1,
                metadata={'seed': seed, 'hypothesis': hypothesis.to_dict()}
            )
            
            # Run experiment
            experiment = IntegratedExperiment(config, self.campaign_dir)
            results = experiment.run()
            
            # Convert to ExperimentData for analysis
            exp_data = ExperimentData(
                name=f"{hypothesis.id}_seed{seed}",
                metrics_history=experiment.metrics_history,
                communication_events=experiment.collected_events[-1000:],  # Last 1000 events
                episode_rewards=experiment.episode_rewards,
                agent_trajectories=experiment.agent_trajectories,
                emergence_timestep=results.get('emergence_timestep'),
                metadata=results
            )
            
            experiment_results.append(exp_data)
            
        return experiment_results
        
    def run_comparative_study(self,
                            campaign_name: str,
                            hypotheses_to_test: Optional[List[str]] = None,
                            num_seeds: int = 3):
        """Run comparative study across multiple hypotheses."""
        print(f"🚀 Starting Research Campaign: {campaign_name}")
        print("=" * 60)
        
        # Create campaign directory
        self.campaign_dir = self.output_base / campaign_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate hypotheses
        all_hypotheses = (
            HypothesisGenerator.generate_ant_colony_hypotheses() +
            HypothesisGenerator.generate_bee_swarm_hypotheses() +
            HypothesisGenerator.generate_neural_hypotheses()
        )
        
        # Filter if specific hypotheses requested
        if hypotheses_to_test:
            hypotheses = [h for h in all_hypotheses if h.id in hypotheses_to_test]
        else:
            hypotheses = all_hypotheses[:3]  # Default to first 3
            
        # Test each hypothesis
        all_results = {}
        
        for hypothesis in hypotheses:
            experiment_results = self.run_hypothesis_test(hypothesis, num_seeds)
            all_results[hypothesis.id] = experiment_results
            
        # Generate comprehensive analysis
        self._generate_comparative_analysis(all_results, campaign_name)
        
        print(f"\n✅ Campaign complete! Results in: {self.campaign_dir}")
        
    def _generate_comparative_analysis(self, 
                                     results: Dict[str, List[ExperimentData]], 
                                     campaign_name: str):
        """Generate comparative analysis across hypotheses."""
        print("\n📊 Generating comparative analysis...")
        
        # Flatten results for overall analysis
        all_experiments = []
        for hypothesis_id, exp_list in results.items():
            for exp in exp_list:
                exp.metadata['hypothesis_id'] = hypothesis_id
                all_experiments.append(exp)
                
        # Create analysis report
        reporter = ExperimentReporter(self.campaign_dir)
        reporter.generate_full_report(all_experiments, f"{campaign_name}_analysis")
        
        # Additional comparative visualizations
        visualizer = CommunicationVisualizer()
        analyzer = EmergenceAnalyzer()
        
        # Compare emergence times across hypotheses
        emergence_comparison = {}
        for hypothesis_id, experiments in results.items():
            emergence_times = [exp.emergence_timestep for exp in experiments 
                             if exp.emergence_timestep is not None]
            if emergence_times:
                emergence_comparison[hypothesis_id] = {
                    'mean': np.mean(emergence_times),
                    'std': np.std(emergence_times),
                    'success_rate': len(emergence_times) / len(experiments)
                }
                
        # Save comparison
        with open(self.campaign_dir / "emergence_comparison.json", 'w') as f:
            json.dump(emergence_comparison, f, indent=2)
            
        # Create summary plot
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Emergence times
        hypotheses = list(emergence_comparison.keys())
        means = [emergence_comparison[h]['mean'] for h in hypotheses]
        stds = [emergence_comparison[h]['std'] for h in hypotheses]
        
        ax1.bar(hypotheses, means, yerr=stds, capsize=5)
        ax1.set_ylabel('Mean Emergence Time (steps)')
        ax1.set_title('Communication Emergence Speed')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Success rates
        success_rates = [emergence_comparison[h]['success_rate'] * 100 for h in hypotheses]
        
        ax2.bar(hypotheses, success_rates)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Emergence Success Rate')
        ax2.set_ylim(0, 105)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.campaign_dir / "hypothesis_comparison.png", dpi=300)
        plt.close()
        
    def _get_environment_for_pressure(self, pressure: EmergencePressure) -> str:
        """Map pressure to environment type."""
        mapping = {
            EmergencePressure.RESOURCE_SCARCITY: "ForagingEnvironment",
            EmergencePressure.PREDATION: "PredatorPreyEnvironment",
            EmergencePressure.TEMPORAL_COORDINATION: "TemporalCoordinationEnvironment",
            EmergencePressure.INFORMATION_ASYMMETRY: "InformationAsymmetryEnvironment"
        }
        return mapping.get(pressure, "ForagingEnvironment")
        
    def _get_channels_for_paradigm(self, paradigm: CommunicationParadigm) -> Dict[str, Any]:
        """Map paradigm to communication channel configuration."""
        configs = {
            CommunicationParadigm.PHEROMONE: {
                "comm_channels": 2,  # Two pheromone types
                "channel_type": "spatial",
                "decay_rate": 0.95
            },
            CommunicationParadigm.SYMBOLIC: {
                "comm_channels": 8,  # Discrete symbols
                "channel_type": "discrete",
                "vocab_size": 16
            },
            CommunicationParadigm.CONTINUOUS: {
                "comm_channels": 4,  # Continuous signals
                "channel_type": "continuous",
                "bandwidth": 1.0
            }
        }
        return configs.get(paradigm, {"comm_channels": 8})


def main():
    """Main entry point for research experiments."""
    parser = argparse.ArgumentParser(
        description="Run nature-inspired MARL communication research experiments"
    )
    
    parser.add_argument(
        "--experiment",
        choices=["ant_foraging", "bee_dance", "predator_prey", "all", "custom"],
        default="ant_foraging",
        help="Which experiment to run"
    )
    
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of random seeds per experiment"
    )
    
    parser.add_argument(
        "--training_steps",
        type=int,
        default=5000,
        help="Training steps per experiment"
    )
    
    parser.add_argument(
        "--campaign_name",
        type=str,
        default="communication_emergence_study",
        help="Name for the research campaign"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(local_mode=True)  # Use local mode for debugging
    
    # Create campaign runner
    runner = ResearchCampaignRunner()
    
    # Run experiments based on selection
    if args.experiment == "all":
        # Run comparative study across multiple hypotheses
        runner.run_comparative_study(
            args.campaign_name,
            num_seeds=args.num_seeds
        )
        
    elif args.experiment == "custom":
        # Define custom hypothesis
        custom_hypothesis = ResearchHypothesis(
            id="custom_mixed_paradigm",
            description="Mixed pheromone and symbolic communication in foraging",
            paradigm=CommunicationParadigm.MULTIMODAL,
            pressure=EmergencePressure.RESOURCE_SCARCITY,
            predictions=[
                "Agents will use pheromones for trails and symbols for resource quality",
                "Communication efficiency will exceed single-paradigm approaches"
            ],
            metrics=["mutual_information", "foraging_efficiency"],
            parameters={
                "pheromone_channels": 1,
                "symbolic_channels": 4
            }
        )
        
        results = runner.run_hypothesis_test(
            custom_hypothesis,
            num_seeds=args.num_seeds,
            training_steps=args.training_steps
        )
        
        # Generate report
        reporter = ExperimentReporter(runner.campaign_dir)
        reporter.generate_full_report(results, "custom_hypothesis_analysis")
        
    else:
        # Run single hypothesis test
        hypothesis_map = {
            "ant_foraging": "ant_pheromone_foraging",
            "bee_dance": "bee_waggle_dance",
            "predator_prey": "ant_danger_signals"
        }
        
        # Get hypothesis
        all_hypotheses = (
            HypothesisGenerator.generate_ant_colony_hypotheses() +
            HypothesisGenerator.generate_bee_swarm_hypotheses()
        )
        
        hypothesis = next(h for h in all_hypotheses 
                         if h.id == hypothesis_map[args.experiment])
        
        # Create campaign directory
        runner.campaign_dir = runner.output_base / args.campaign_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        runner.campaign_dir.mkdir(parents=True, exist_ok=True)
        
        # Run experiment
        results = runner.run_hypothesis_test(
            hypothesis,
            num_seeds=args.num_seeds,
            training_steps=args.training_steps
        )
        
        # Generate report
        reporter = ExperimentReporter(runner.campaign_dir)
        reporter.generate_full_report(results, f"{args.experiment}_analysis")
    
    # Shutdown Ray
    ray.shutdown()
    
    print("\n🎉 Research experiment complete!")
    print(f"📁 Results saved to: {runner.campaign_dir}")


if __name__ == "__main__":
    main()
