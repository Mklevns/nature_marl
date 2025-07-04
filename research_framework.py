# File: marlcomm/research_framework.py
"""
Research Framework for Studying Emergent Communication in Multi-Agent Systems

This module provides the scientific foundation for investigating how efficient
communication networks emerge in multi-agent reinforcement learning environments.
It manages experiments, tracks hypotheses, and ensures reproducible research.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from enum import Enum


class CommunicationParadigm(Enum):
    """Types of communication paradigms to study."""
    PHEROMONE = "pheromone"  # Chemical trails that decay over time
    SYMBOLIC = "symbolic"  # Discrete symbol exchange
    CONTINUOUS = "continuous"  # Continuous signal transmission
    SPATIAL = "spatial"  # Location-based signaling
    MULTIMODAL = "multimodal"  # Combination of multiple channels


class EmergencePressure(Enum):
    """Environmental pressures that might lead to communication emergence."""
    RESOURCE_SCARCITY = "resource_scarcity"
    PREDATION = "predation"
    INFORMATION_ASYMMETRY = "information_asymmetry"
    TEMPORAL_COORDINATION = "temporal_coordination"
    SPATIAL_NAVIGATION = "spatial_navigation"
    COLLECTIVE_CONSTRUCTION = "collective_construction"


@dataclass
class ResearchHypothesis:
    """Structured representation of a research hypothesis."""
    id: str
    description: str
    paradigm: CommunicationParadigm
    pressure: EmergencePressure
    predictions: List[str]
    metrics: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "paradigm": self.paradigm.value,
            "pressure": self.pressure.value,
            "predictions": self.predictions,
            "metrics": self.metrics,
            "parameters": self.parameters
        }


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    hypothesis_id: str
    environment_type: str
    agent_architecture: str
    communication_channels: Dict[str, Any]
    reward_structure: Dict[str, Any]
    training_steps: int
    num_agents: int
    num_seeds: int = 3
    ablation_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_unique_id(self) -> str:
        """Generate unique experiment ID based on configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class EmergenceMetrics:
    """Metrics for measuring communication emergence."""
    # Information-theoretic metrics
    mutual_information: float = 0.0
    channel_capacity: float = 0.0
    entropy_rate: float = 0.0
    
    # Behavioral metrics
    coordination_efficiency: float = 0.0
    task_success_rate: float = 0.0
    response_latency: float = 0.0
    
    # Network metrics
    communication_frequency: float = 0.0
    network_density: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Complexity metrics
    signal_complexity: float = 0.0
    protocol_stability: float = 0.0
    adaptation_rate: float = 0.0


class EmergenceDetector:
    """Detects when communication patterns have emerged."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        
    def update(self, metrics: EmergenceMetrics) -> bool:
        """
        Update detector with new metrics and check for emergence.
        
        Returns:
            True if communication has emerged based on metrics
        """
        self.history.append(metrics)
        
        if len(self.history) < self.window_size:
            return False
            
        # Check multiple criteria for emergence
        recent_metrics = self.history[-self.window_size:]
        
        # Criterion 1: Stable non-zero communication
        avg_comm_freq = np.mean([m.communication_frequency for m in recent_metrics])
        comm_variance = np.var([m.communication_frequency for m in recent_metrics])
        stable_communication = avg_comm_freq > 0.1 and comm_variance < 0.1
        
        # Criterion 2: Improved coordination
        early_coordination = np.mean([m.coordination_efficiency 
                                     for m in self.history[:self.window_size//2]])
        recent_coordination = np.mean([m.coordination_efficiency 
                                      for m in recent_metrics])
        improved_coordination = recent_coordination > early_coordination * 1.2
        
        # Criterion 3: Information flow
        avg_mutual_info = np.mean([m.mutual_information for m in recent_metrics])
        meaningful_info = avg_mutual_info > 0.1
        
        return stable_communication and improved_coordination and meaningful_info


class ResearchExperiment(ABC):
    """Base class for communication emergence experiments."""
    
    def __init__(self, config: ExperimentConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.experiment_id = config.get_unique_id()
        self.start_time = datetime.now()
        self.emergence_detector = EmergenceDetector()
        self.results = {
            "config": asdict(config),
            "metrics_history": [],
            "emergence_timestep": None,
            "final_metrics": None
        }
        
    @abstractmethod
    def setup_environment(self) -> Any:
        """Create and configure the environment."""
        pass
    
    @abstractmethod
    def setup_agents(self) -> Any:
        """Initialize agent policies and communication modules."""
        pass
    
    @abstractmethod
    def run_training_step(self) -> EmergenceMetrics:
        """Execute one training iteration and return metrics."""
        pass
    
    @abstractmethod
    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze emerged communication patterns."""
        pass
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete experiment."""
        print(f"🧪 Starting experiment {self.experiment_id}")
        print(f"   Hypothesis: {self.config.hypothesis_id}")
        print(f"   Environment: {self.config.environment_type}")
        
        # Setup
        self.setup_environment()
        self.setup_agents()
        
        # Training loop
        for step in range(self.config.training_steps):
            metrics = self.run_training_step()
            self.results["metrics_history"].append(asdict(metrics))
            
            # Check for emergence
            if self.emergence_detector.update(metrics):
                if self.results["emergence_timestep"] is None:
                    self.results["emergence_timestep"] = step
                    print(f"🎉 Communication emerged at step {step}!")
            
            # Progress logging
            if step % 100 == 0:
                print(f"   Step {step}/{self.config.training_steps} - "
                      f"Coordination: {metrics.coordination_efficiency:.3f}, "
                      f"MI: {metrics.mutual_information:.3f}")
        
        # Final analysis
        self.results["final_metrics"] = self.results["metrics_history"][-1]
        self.results["communication_analysis"] = self.analyze_communication_patterns()
        self.results["duration"] = (datetime.now() - self.start_time).total_seconds()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save experiment results to disk."""
        results_path = self.output_dir / f"experiment_{self.experiment_id}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to {results_path}")


class HypothesisGenerator:
    """Generates research hypotheses based on biological inspiration."""
    
    @staticmethod
    def generate_ant_colony_hypotheses() -> List[ResearchHypothesis]:
        """Generate hypotheses inspired by ant colony communication."""
        return [
            ResearchHypothesis(
                id="ant_pheromone_foraging",
                description="Pheromone trails emerge when agents face distributed "
                           "resources with decay, similar to ant foraging",
                paradigm=CommunicationParadigm.PHEROMONE,
                pressure=EmergencePressure.RESOURCE_SCARCITY,
                predictions=[
                    "Agents will develop trail-following behavior",
                    "Shorter paths will be reinforced through positive feedback",
                    "Communication efficiency will scale with colony size"
                ],
                metrics=["trail_coherence", "path_optimality", "foraging_efficiency"],
                parameters={
                    "pheromone_decay_rate": 0.95,
                    "pheromone_deposit_rate": 1.0,
                    "resource_distribution": "clustered"
                }
            ),
            ResearchHypothesis(
                id="ant_danger_signals",
                description="Alarm pheromones emerge under predation pressure",
                paradigm=CommunicationParadigm.PHEROMONE,
                pressure=EmergencePressure.PREDATION,
                predictions=[
                    "Distinct alarm signals will differentiate from trail pheromones",
                    "Alarm response will propagate through the colony",
                    "False alarm rate will decrease over time"
                ],
                metrics=["alarm_specificity", "response_time", "survival_rate"],
                parameters={
                    "predator_frequency": 0.1,
                    "alarm_pheromone_intensity": 5.0,
                    "alarm_decay_rate": 0.8
                }
            )
        ]
    
    @staticmethod
    def generate_bee_swarm_hypotheses() -> List[ResearchHypothesis]:
        """Generate hypotheses inspired by bee swarm communication."""
        return [
            ResearchHypothesis(
                id="bee_waggle_dance",
                description="Symbolic communication emerges for distant resource location",
                paradigm=CommunicationParadigm.SYMBOLIC,
                pressure=EmergencePressure.INFORMATION_ASYMMETRY,
                predictions=[
                    "Agents will encode distance and direction in signals",
                    "Signal complexity will correlate with resource value",
                    "Recruitment efficiency will improve over time"
                ],
                metrics=["signal_accuracy", "recruitment_rate", "decoding_success"],
                parameters={
                    "resource_distance_range": [10, 100],
                    "signal_dimensions": 8,
                    "observation_noise": 0.1
                }
            )
        ]
    
    @staticmethod
    def generate_neural_hypotheses() -> List[ResearchHypothesis]:
        """Generate hypotheses inspired by neural communication."""
        return [
            ResearchHypothesis(
                id="neural_synchronization",
                description="Temporal synchronization emerges for collective decision-making",
                paradigm=CommunicationParadigm.CONTINUOUS,
                pressure=EmergencePressure.TEMPORAL_COORDINATION,
                predictions=[
                    "Agents will develop synchronized firing patterns",
                    "Phase locking will correlate with task performance",
                    "Network will self-organize into functional modules"
                ],
                metrics=["synchronization_index", "phase_coherence", "modularity"],
                parameters={
                    "coupling_strength": 0.5,
                    "intrinsic_frequency_variance": 0.1,
                    "network_topology": "small_world"
                }
            )
        ]


class ExperimentOrchestrator:
    """Manages multiple experiments and research campaigns."""
    
    def __init__(self, output_dir: Path = Path("research_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.hypotheses = []
        self.experiments = []
        
    def add_hypothesis(self, hypothesis: ResearchHypothesis):
        """Add a research hypothesis to test."""
        self.hypotheses.append(hypothesis)
        
    def generate_experiment_configs(self, hypothesis: ResearchHypothesis, 
                                  variations: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Generate experiment configurations for hypothesis testing."""
        configs = []
        
        # Base configuration
        base_config = ExperimentConfig(
            hypothesis_id=hypothesis.id,
            environment_type=self._get_environment_for_pressure(hypothesis.pressure),
            agent_architecture="NatureInspiredCommModule",
            communication_channels=self._get_channels_for_paradigm(hypothesis.paradigm),
            reward_structure=self._get_reward_for_pressure(hypothesis.pressure),
            training_steps=10000,
            num_agents=5
        )
        
        # Generate variations for ablation studies
        for param, values in variations.items():
            for value in values:
                config = ExperimentConfig(**asdict(base_config))
                config.ablation_params[param] = value
                configs.append(config)
                
        return configs
    
    def _get_environment_for_pressure(self, pressure: EmergencePressure) -> str:
        """Map emergence pressure to environment type."""
        mapping = {
            EmergencePressure.RESOURCE_SCARCITY: "ForagingEnvironment",
            EmergencePressure.PREDATION: "PredatorPreyEnvironment",
            EmergencePressure.INFORMATION_ASYMMETRY: "PartialInfoEnvironment",
            EmergencePressure.TEMPORAL_COORDINATION: "SynchronizationEnvironment",
            EmergencePressure.SPATIAL_NAVIGATION: "MazeNavigationEnvironment",
            EmergencePressure.COLLECTIVE_CONSTRUCTION: "ConstructionEnvironment"
        }
        return mapping.get(pressure, "SimpleSpreadEnvironment")
    
    def _get_channels_for_paradigm(self, paradigm: CommunicationParadigm) -> Dict[str, Any]:
        """Configure communication channels for paradigm."""
        configs = {
            CommunicationParadigm.PHEROMONE: {
                "type": "spatial_field",
                "decay_rate": 0.95,
                "diffusion_rate": 0.1,
                "max_intensity": 10.0
            },
            CommunicationParadigm.SYMBOLIC: {
                "type": "discrete_symbols", 
                "vocabulary_size": 16,
                "max_message_length": 4
            },
            CommunicationParadigm.CONTINUOUS: {
                "type": "continuous_broadcast",
                "channel_dim": 8,
                "broadcast_range": 5.0,
                "noise_level": 0.01
            }
        }
        return configs.get(paradigm, {})
    
    def _get_reward_for_pressure(self, pressure: EmergencePressure) -> Dict[str, Any]:
        """Design reward structure for emergence pressure."""
        rewards = {
            EmergencePressure.RESOURCE_SCARCITY: {
                "individual_collection": 1.0,
                "collective_efficiency": 2.0,
                "energy_cost": -0.1
            },
            EmergencePressure.PREDATION: {
                "survival": 10.0,
                "collective_defense": 5.0,
                "false_alarm_cost": -1.0
            }
        }
        return rewards.get(pressure, {"task_completion": 1.0})
    
    def run_research_campaign(self, campaign_name: str, 
                            hypothesis_selector: Callable = None) -> Dict[str, Any]:
        """Run a complete research campaign testing multiple hypotheses."""
        print(f"🚀 Starting research campaign: {campaign_name}")
        
        # Generate hypotheses
        all_hypotheses = (
            HypothesisGenerator.generate_ant_colony_hypotheses() +
            HypothesisGenerator.generate_bee_swarm_hypotheses() +
            HypothesisGenerator.generate_neural_hypotheses()
        )
        
        # Filter hypotheses if selector provided
        if hypothesis_selector:
            hypotheses_to_test = [h for h in all_hypotheses if hypothesis_selector(h)]
        else:
            hypotheses_to_test = all_hypotheses
            
        campaign_results = {
            "campaign_name": campaign_name,
            "start_time": datetime.now().isoformat(),
            "hypotheses_tested": len(hypotheses_to_test),
            "results": []
        }
        
        # Test each hypothesis
        for hypothesis in hypotheses_to_test:
            print(f"\n📊 Testing hypothesis: {hypothesis.id}")
            
            # Generate experiment variations
            variations = {
                "num_agents": [3, 5, 10],
                "communication_noise": [0.0, 0.1, 0.3]
            }
            
            configs = self.generate_experiment_configs(hypothesis, variations)
            
            hypothesis_results = {
                "hypothesis": hypothesis.to_dict(),
                "experiments": []
            }
            
            # Run experiments
            for config in configs:
                # This would instantiate the actual experiment class
                # For now, we'll store the config
                hypothesis_results["experiments"].append({
                    "config": asdict(config),
                    "status": "pending"
                })
                
            campaign_results["results"].append(hypothesis_results)
        
        # Save campaign summary
        campaign_path = self.output_dir / f"campaign_{campaign_name}.json"
        with open(campaign_path, 'w') as f:
            json.dump(campaign_results, f, indent=2)
            
        print(f"\n✅ Campaign completed! Results saved to {campaign_path}")
        return campaign_results


# Example usage and research questions
if __name__ == "__main__":
    # Create research orchestrator
    orchestrator = ExperimentOrchestrator()
    
    # Run a focused campaign on pheromone communication
    results = orchestrator.run_research_campaign(
        "pheromone_emergence_study",
        hypothesis_selector=lambda h: h.paradigm == CommunicationParadigm.PHEROMONE
    )
    
    print("\n🔬 Research Questions to Explore:")
    print("1. How does environmental complexity affect communication emergence?")
    print("2. What is the minimum channel capacity for efficient coordination?")
    print("3. How do different network topologies influence protocol evolution?")
    print("4. Can agents discover error correction without explicit training?")
    print("5. What role does memory play in communication stability?")
