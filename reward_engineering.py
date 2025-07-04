# File: marlcomm/reward_engineering.py
"""
Reward Engineering Module for Communication Emergence

This module provides sophisticated reward design techniques that implicitly
encourage communication emergence without explicitly rewarding message passing.
The goal is to create environments where communication naturally becomes
the optimal strategy for maximizing rewards.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import torch
import torch.nn.functional as F
from collections import defaultdict, deque


class RewardPrinciple(Enum):
    """Core principles for implicit communication rewards."""
    INFORMATION_ASYMMETRY = "information_asymmetry"  # Reward using private info
    TEMPORAL_COUPLING = "temporal_coupling"  # Reward synchronized actions
    SPATIAL_COORDINATION = "spatial_coordination"  # Reward spatial patterns
    COLLECTIVE_EFFICIENCY = "collective_efficiency"  # Reward group performance
    EMERGENT_SPECIALIZATION = "emergent_specialization"  # Reward role differentiation
    ADAPTIVE_CHALLENGE = "adaptive_challenge"  # Increase difficulty with capability


@dataclass
class RewardContext:
    """Context information for reward calculation."""
    agent_states: Dict[str, np.ndarray]
    agent_actions: Dict[str, np.ndarray]
    agent_observations: Dict[str, np.ndarray]
    communication_signals: Dict[str, np.ndarray]
    environmental_state: Dict[str, Any]
    timestep: int
    episode_history: List[Dict[str, Any]] = field(default_factory=list)


class RewardComponent(ABC):
    """Base class for modular reward components."""

    def __init__(self, weight: float = 1.0, warmup_steps: int = 0):
        self.weight = weight
        self.warmup_steps = warmup_steps
        self.step_count = 0

    @abstractmethod
    def calculate(self, agent_id: str, context: RewardContext) -> float:
        """Calculate reward component for an agent."""
        pass

    @abstractmethod
    def get_principle(self) -> RewardPrinciple:
        """Get the principle this component implements."""
        pass

    def get_current_weight(self) -> float:
        """Get weight with warmup scheduling."""
        if self.step_count < self.warmup_steps:
            return self.weight * (self.step_count / self.warmup_steps)
        return self.weight

    def update(self):
        """Update internal state."""
        self.step_count += 1


class InformationAsymmetryReward(RewardComponent):
    """
    Reward agents for achieving goals that require information from others.
    Creates pressure to share private observations.
    """

    def __init__(self, info_value_fn: Callable[[str, Dict], float], **kwargs):
        super().__init__(**kwargs)
        self.info_value_fn = info_value_fn
        self.information_usage_history = defaultdict(list)

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        # Check if agent used information it couldn't directly observe
        own_obs = context.agent_observations[agent_id]

        # Estimate what information was used based on action
        action = context.agent_actions[agent_id]

        # Calculate value of information that could only come from others
        info_value = self.info_value_fn(agent_id, {
            'action': action,
            'observation': own_obs,
            'environment': context.environmental_state
        })

        # Track information usage
        self.information_usage_history[agent_id].append(info_value)

        # Reward is based on using non-local information effectively
        reward = info_value * self.get_current_weight()

        return reward

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.INFORMATION_ASYMMETRY


class TemporalCouplingReward(RewardComponent):
    """
    Reward synchronized actions without explicit coordination rewards.
    Creates pressure for timing communication.
    """

    def __init__(self, sync_window: int = 5, sync_threshold: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.sync_window = sync_window
        self.sync_threshold = sync_threshold
        self.action_history = defaultdict(lambda: deque(maxlen=sync_window))

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        # Record action
        self.action_history[agent_id].append(context.agent_actions[agent_id])

        # Check synchronization with other agents
        if len(context.agent_actions) < 2:
            return 0.0

        sync_scores = []
        for other_id, other_action in context.agent_actions.items():
            if other_id != agent_id:
                # Calculate action similarity
                similarity = self._calculate_action_similarity(
                    context.agent_actions[agent_id],
                    other_action
                )
                sync_scores.append(similarity)

        # Check if synchronized actions led to success
        avg_sync = np.mean(sync_scores) if sync_scores else 0.0

        # Reward only if synchronization achieves environmental goal
        env_success = context.environmental_state.get('sync_goal_achieved', False)

        if avg_sync > self.sync_threshold and env_success:
            return 1.0 * self.get_current_weight()

        return 0.0

    def _calculate_action_similarity(self, action1: np.ndarray, action2: np.ndarray) -> float:
        """Calculate similarity between two actions."""
        if action1.shape != action2.shape:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(action1.flatten(), action2.flatten())
        norm_product = np.linalg.norm(action1) * np.linalg.norm(action2)

        if norm_product > 0:
            return (dot_product / norm_product + 1) / 2  # Normalize to [0, 1]
        return 0.5

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.TEMPORAL_COUPLING


class CollectiveEfficiencyReward(RewardComponent):
    """
    Reward group performance improvements over individual baselines.
    Communication should enable better collective outcomes.
    """

    def __init__(self, individual_baseline_fn: Callable, efficiency_threshold: float = 1.2, **kwargs):
        super().__init__(**kwargs)
        self.individual_baseline_fn = individual_baseline_fn
        self.efficiency_threshold = efficiency_threshold
        self.performance_history = defaultdict(lambda: deque(maxlen=100))

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        # Calculate individual baseline (what agent could achieve alone)
        individual_baseline = self.individual_baseline_fn(agent_id, context)

        # Calculate actual collective performance
        collective_performance = self._calculate_collective_performance(context)

        # Track performance
        self.performance_history['individual'].append(individual_baseline)
        self.performance_history['collective'].append(collective_performance)

        # Reward if collective performance exceeds individual baseline significantly
        if collective_performance > individual_baseline * self.efficiency_threshold:
            efficiency_bonus = (collective_performance / individual_baseline - 1.0)
            return efficiency_bonus * self.get_current_weight()

        return 0.0

    def _calculate_collective_performance(self, context: RewardContext) -> float:
        """Calculate group's collective performance metric."""
        # Example: Total resources collected, area covered, etc.
        total_performance = context.environmental_state.get('collective_score', 0.0)
        return total_performance / max(len(context.agent_states), 1)

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.COLLECTIVE_EFFICIENCY


class EmergentSpecializationReward(RewardComponent):
    """
    Reward role differentiation that emerges through communication.
    Agents should discover complementary behaviors.
    """

    def __init__(self, n_roles: int = 3, diversity_bonus: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_roles = n_roles
        self.diversity_bonus = diversity_bonus
        self.behavior_embeddings = defaultdict(lambda: deque(maxlen=50))

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        # Extract behavior embedding from action and state
        behavior = self._extract_behavior_embedding(agent_id, context)
        self.behavior_embeddings[agent_id].append(behavior)

        # Calculate role diversity across agents
        if len(self.behavior_embeddings) >= 2:
            diversity = self._calculate_behavioral_diversity()

            # Reward if agent contributes to diversity while being consistent
            consistency = self._calculate_behavioral_consistency(agent_id)

            if diversity > 0.5 and consistency > 0.7:
                return (diversity * consistency * self.diversity_bonus *
                       self.get_current_weight())

        return 0.0

    def _extract_behavior_embedding(self, agent_id: str, context: RewardContext) -> np.ndarray:
        """Extract behavioral features for role identification."""
        action = context.agent_actions[agent_id]
        state = context.agent_states[agent_id]

        # Combine action patterns and state preferences
        behavior = np.concatenate([
            action.flatten()[:5],  # Action tendencies
            state.flatten()[:5],   # State preferences
        ])

        return behavior / (np.linalg.norm(behavior) + 1e-8)

    def _calculate_behavioral_diversity(self) -> float:
        """Calculate diversity of behaviors across agents."""
        if len(self.behavior_embeddings) < 2:
            return 0.0

        # Get recent behaviors for each agent
        recent_behaviors = []
        for agent_behaviors in self.behavior_embeddings.values():
            if len(agent_behaviors) > 0:
                recent_behaviors.append(np.mean(agent_behaviors, axis=0))

        if len(recent_behaviors) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(recent_behaviors)):
            for j in range(i + 1, len(recent_behaviors)):
                dist = np.linalg.norm(recent_behaviors[i] - recent_behaviors[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _calculate_behavioral_consistency(self, agent_id: str) -> float:
        """Calculate consistency of agent's behavior over time."""
        behaviors = list(self.behavior_embeddings[agent_id])
        if len(behaviors) < 5:
            return 0.0

        # Calculate variance in behavior
        behaviors_array = np.array(behaviors)
        variance = np.mean(np.var(behaviors_array, axis=0))

        # Convert to consistency score (low variance = high consistency)
        consistency = np.exp(-variance * 5)

        return consistency

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.EMERGENT_SPECIALIZATION


class AdaptiveChallengeReward(RewardComponent):
    """
    Dynamically adjust task difficulty based on communication competence.
    Maintains pressure for continued communication development.
    """

    def __init__(self, base_difficulty: float = 0.5, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.base_difficulty = base_difficulty
        self.current_difficulty = base_difficulty
        self.adaptation_rate = adaptation_rate
        self.success_history = deque(maxlen=100)

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        # Check if task was completed
        task_success = context.environmental_state.get('task_completed', False)
        self.success_history.append(float(task_success))

        # Adapt difficulty based on success rate
        if len(self.success_history) > 20:
            success_rate = np.mean(self.success_history)

            # Increase difficulty if too successful, decrease if struggling
            if success_rate > 0.8:
                self.current_difficulty = min(1.0, self.current_difficulty + self.adaptation_rate)
            elif success_rate < 0.3:
                self.current_difficulty = max(0.1, self.current_difficulty - self.adaptation_rate)

        # Reward scales with difficulty
        if task_success:
            return self.current_difficulty * self.get_current_weight()

        return 0.0

    def update_environment_difficulty(self, env_state: Dict[str, Any]):
        """Update environment parameters based on current difficulty."""
        env_state['difficulty_multiplier'] = self.current_difficulty
        env_state['n_obstacles'] = int(5 * self.current_difficulty)
        env_state['time_pressure'] = 1.0 - self.current_difficulty * 0.5

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.ADAPTIVE_CHALLENGE


class CounterfactualCommunicationReward(RewardComponent):
    """
    Estimate the value of communication by comparing with counterfactual
    scenarios where communication didn't occur.
    """

    def __init__(self, counterfactual_model: Optional[Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.counterfactual_model = counterfactual_model
        self.value_estimates = defaultdict(list)

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        # Estimate outcome without communication
        no_comm_outcome = self._estimate_no_communication_outcome(agent_id, context)

        # Actual outcome with communication
        actual_outcome = context.environmental_state.get(f'{agent_id}_outcome', 0.0)

        # Communication value is the difference
        comm_value = actual_outcome - no_comm_outcome

        self.value_estimates[agent_id].append(comm_value)

        # Only reward if communication provided clear benefit
        if comm_value > 0.1:
            return comm_value * self.get_current_weight()

        return 0.0

    def _estimate_no_communication_outcome(self, agent_id: str, context: RewardContext) -> float:
        """Estimate what would happen without communication."""
        if self.counterfactual_model:
            # Use learned model
            features = self._extract_features(agent_id, context)
            return self.counterfactual_model.predict(features)
        else:
            # Simple heuristic: performance with only local information
            local_info_quality = context.environmental_state.get(f'{agent_id}_local_info', 0.5)
            return local_info_quality * 0.5  # Baseline performance

    def _extract_features(self, agent_id: str, context: RewardContext) -> np.ndarray:
        """Extract features for counterfactual prediction."""
        return np.concatenate([
            context.agent_observations[agent_id].flatten(),
            context.agent_states[agent_id].flatten()
        ])

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.INFORMATION_ASYMMETRY


class SocialDilemmaReward(RewardComponent):
    """
    Create social dilemmas that require communication to resolve.
    E.g., situations where individual and collective interests conflict.
    """

    def __init__(self, dilemma_type: str = "public_goods", cooperation_multiplier: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.dilemma_type = dilemma_type
        self.cooperation_multiplier = cooperation_multiplier
        self.cooperation_history = defaultdict(lambda: deque(maxlen=50))

    def calculate(self, agent_id: str, context: RewardContext) -> float:
        if self.dilemma_type == "public_goods":
            return self._public_goods_game(agent_id, context)
        elif self.dilemma_type == "coordination":
            return self._coordination_game(agent_id, context)
        elif self.dilemma_type == "trust":
            return self._trust_game(agent_id, context)
        else:
            return 0.0

    def _public_goods_game(self, agent_id: str, context: RewardContext) -> float:
        """
        Public goods game where agents choose to contribute or free-ride.
        Communication enables agreements and reputation tracking.
        """
        # Check agent's contribution
        contribution = context.environmental_state.get(f'{agent_id}_contribution', 0.0)
        total_contributions = context.environmental_state.get('total_contributions', 0.0)
        n_agents = len(context.agent_states)

        # Public good benefit (shared equally)
        public_benefit = (total_contributions * self.cooperation_multiplier) / n_agents

        # Individual cost of contribution
        individual_cost = contribution

        # Net reward
        base_reward = public_benefit - individual_cost

        # Bonus for achieving high collective contribution through communication
        if total_contributions > n_agents * 0.7:  # 70% contribution rate
            coordination_bonus = 0.5 * self.get_current_weight()
            return base_reward + coordination_bonus

        return base_reward

    def _coordination_game(self, agent_id: str, context: RewardContext) -> float:
        """
        Coordination game where agents must choose matching strategies.
        Communication enables strategy agreement.
        """
        agent_choice = context.agent_actions[agent_id][0]  # First action dimension

        # Count how many others made the same choice
        matching_agents = 0
        for other_id, other_action in context.agent_actions.items():
            if other_id != agent_id and np.abs(other_action[0] - agent_choice) < 0.1:
                matching_agents += 1

        # Reward for coordination
        coordination_reward = matching_agents / max(len(context.agent_actions) - 1, 1)

        return coordination_reward * self.get_current_weight()

    def _trust_game(self, agent_id: str, context: RewardContext) -> float:
        """
        Trust game where agents must rely on others' promises.
        Communication enables trust building.
        """
        # Check if agent trusted (took risky cooperative action)
        trusted = context.environmental_state.get(f'{agent_id}_trusted', False)
        was_betrayed = context.environmental_state.get(f'{agent_id}_betrayed', False)

        if trusted and not was_betrayed:
            return 2.0 * self.get_current_weight()  # High reward for successful trust
        elif trusted and was_betrayed:
            return -1.0  # Penalty for misplaced trust
        else:
            return 0.2  # Small safe reward

    def get_principle(self) -> RewardPrinciple:
        return RewardPrinciple.COLLECTIVE_EFFICIENCY


class RewardEngineer:
    """
    Main class for engineering reward functions that implicitly encourage communication.
    Combines multiple reward components with curriculum learning.
    """

    def __init__(self, environment_type: str):
        self.environment_type = environment_type
        self.reward_components: List[RewardComponent] = []
        self.reward_history = defaultdict(lambda: defaultdict(list))
        self.curriculum_stage = 0
        self.stage_transitions = []

    def add_component(self, component: RewardComponent):
        """Add a reward component to the engineer."""
        self.reward_components.append(component)

    def create_implicit_reward_structure(self, paradigm: str = "multi_principle") -> Dict[str, Any]:
        """
        Create a complete implicit reward structure based on paradigm.

        Paradigms:
        - 'multi_principle': Combine multiple principles
        - 'curriculum': Gradually introduce complexity
        - 'adversarial': Adapt to prevent reward hacking
        """
        if paradigm == "multi_principle":
            return self._create_multi_principle_rewards()
        elif paradigm == "curriculum":
            return self._create_curriculum_rewards()
        elif paradigm == "adversarial":
            return self._create_adversarial_rewards()
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

    def _create_multi_principle_rewards(self) -> Dict[str, Any]:
        """Combine multiple reward principles."""
        # Clear existing components
        self.reward_components.clear()

        # Add complementary components based on environment
        if self.environment_type == "foraging":
            # Information about food locations
            self.add_component(InformationAsymmetryReward(
                info_value_fn=self._foraging_info_value,
                weight=0.3
            ))

            # Collective efficiency in gathering
            self.add_component(CollectiveEfficiencyReward(
                individual_baseline_fn=self._foraging_individual_baseline,
                weight=0.4
            ))

            # Role specialization (scouts vs gatherers)
            self.add_component(EmergentSpecializationReward(
                n_roles=2,
                weight=0.3
            ))

        elif self.environment_type == "predator_prey":
            # Synchronized escape
            self.add_component(TemporalCouplingReward(
                sync_window=3,
                weight=0.5
            ))

            # Information about predator locations
            self.add_component(InformationAsymmetryReward(
                info_value_fn=self._predator_info_value,
                weight=0.5
            ))

        return {
            "components": self.reward_components,
            "combination_method": "weighted_sum"
        }

    def _create_curriculum_rewards(self) -> Dict[str, Any]:
        """Create curriculum that gradually increases communication necessity."""
        curriculum_stages = [
            # Stage 1: Basic task completion
            {
                "duration": 1000,
                "components": [
                    CollectiveEfficiencyReward(
                        individual_baseline_fn=lambda a, c: 0.5,
                        weight=1.0,
                        warmup_steps=0
                    )
                ]
            },
            # Stage 2: Introduce information asymmetry
            {
                "duration": 2000,
                "components": [
                    CollectiveEfficiencyReward(
                        individual_baseline_fn=lambda a, c: 0.3,
                        weight=0.5
                    ),
                    InformationAsymmetryReward(
                        info_value_fn=self._generic_info_value,
                        weight=0.5,
                        warmup_steps=500
                    )
                ]
            },
            # Stage 3: Add specialization pressure
            {
                "duration": 3000,
                "components": [
                    CollectiveEfficiencyReward(weight=0.3),
                    InformationAsymmetryReward(weight=0.3),
                    EmergentSpecializationReward(
                        n_roles=3,
                        weight=0.4,
                        warmup_steps=500
                    )
                ]
            },
            # Stage 4: Adaptive challenge
            {
                "duration": -1,  # Infinite
                "components": [
                    CollectiveEfficiencyReward(weight=0.2),
                    InformationAsymmetryReward(weight=0.2),
                    EmergentSpecializationReward(weight=0.2),
                    AdaptiveChallengeReward(
                        weight=0.4,
                        adaptation_rate=0.005
                    )
                ]
            }
        ]

        return {
            "stages": curriculum_stages,
            "current_stage": self.curriculum_stage,
            "transition_fn": self._check_curriculum_transition
        }

    def _create_adversarial_rewards(self) -> Dict[str, Any]:
        """Create rewards that adapt to prevent exploitation."""
        # Use counterfactual reasoning to ensure communication is truly valuable
        self.reward_components = [
            CounterfactualCommunicationReward(weight=0.5),
            SocialDilemmaReward(
                dilemma_type="public_goods",
                weight=0.5
            )
        ]

        return {
            "components": self.reward_components,
            "adaptation_method": "counterfactual_validation",
            "exploitation_detection": self._detect_reward_hacking
        }

    def calculate_reward(self, agent_id: str, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total reward and component breakdown.

        Returns:
            Tuple of (total_reward, component_rewards)
        """
        component_rewards = {}
        total_reward = 0.0

        for component in self.reward_components:
            component_reward = component.calculate(agent_id, context)
            component_name = component.get_principle().value
            component_rewards[component_name] = component_reward
            total_reward += component_reward

            # Track history
            self.reward_history[agent_id][component_name].append(component_reward)

        # Update components
        for component in self.reward_components:
            component.update()

        return total_reward, component_rewards

    def _check_curriculum_transition(self, total_steps: int, performance_metrics: Dict[str, float]) -> bool:
        """Check if curriculum should advance to next stage."""
        # Simple transition based on steps and performance
        min_performance = performance_metrics.get('success_rate', 0.0)

        if min_performance > 0.7 and total_steps > self.curriculum_stage * 1000:
            self.curriculum_stage += 1
            self.stage_transitions.append({
                'step': total_steps,
                'performance': performance_metrics
            })
            return True

        return False

    def _detect_reward_hacking(self, agent_behaviors: Dict[str, np.ndarray]) -> bool:
        """Detect if agents are exploiting reward function."""
        # Check for suspicious patterns
        for agent_id, behavior in agent_behaviors.items():
            # Repetitive actions without communication
            behavior_variance = np.var(behavior)
            if behavior_variance < 0.01:  # Very low variance
                return True

        return False

    # Helper functions for specific environments
    def _foraging_info_value(self, agent_id: str, info: Dict) -> float:
        """Calculate value of information in foraging context."""
        # Did agent find food in unexpected location?
        action = info['action']
        local_food_visible = info['environment'].get(f'{agent_id}_sees_food', False)
        found_food = info['environment'].get(f'{agent_id}_found_food', False)

        if found_food and not local_food_visible:
            return 1.0  # Used communicated information
        return 0.0

    def _foraging_individual_baseline(self, agent_id: str, context: RewardContext) -> float:
        """Baseline foraging performance without communication."""
        # Random search efficiency
        return 0.2 * context.environmental_state.get('total_food', 1.0)

    def _predator_info_value(self, agent_id: str, info: Dict) -> float:
        """Calculate value of predator warning information."""
        escaped = info['environment'].get(f'{agent_id}_escaped', False)
        was_warned = info['environment'].get(f'{agent_id}_received_warning', False)

        if escaped and was_warned:
            return 1.0
        return 0.0

    def _generic_info_value(self, agent_id: str, info: Dict) -> float:
        """Generic information value calculation."""
        # Simple heuristic: reward if action matches non-visible optimal
        optimal_action = info['environment'].get('optimal_action', None)
        actual_action = info['action']

        if optimal_action is not None:
            similarity = 1.0 - np.linalg.norm(optimal_action - actual_action)
            return max(0, similarity)
        return 0.0

    def analyze_reward_distribution(self) -> Dict[str, Any]:
        """Analyze how rewards are distributed across components and agents."""
        analysis = {
            "component_contributions": {},
            "agent_performance": {},
            "balance_metrics": {}
        }

        # Component contribution analysis
        for component in self.reward_components:
            component_name = component.get_principle().value
            all_rewards = []

            for agent_history in self.reward_history.values():
                if component_name in agent_history:
                    all_rewards.extend(agent_history[component_name])

            if all_rewards:
                analysis["component_contributions"][component_name] = {
                    "mean": np.mean(all_rewards),
                    "std": np.std(all_rewards),
                    "total": np.sum(all_rewards)
                }

        # Agent performance analysis
        for agent_id, history in self.reward_history.items():
            total_rewards = []
            for component_rewards in history.values():
                total_rewards.extend(component_rewards)

            if total_rewards:
                analysis["agent_performance"][agent_id] = {
                    "total": np.sum(total_rewards),
                    "mean": np.mean(total_rewards),
                    "trend": "increasing" if total_rewards[-1] > total_rewards[0] else "decreasing"
                }

        # Balance metrics
        if analysis["agent_performance"]:
            performances = [a["total"] for a in analysis["agent_performance"].values()]
            analysis["balance_metrics"] = {
                "gini_coefficient": self._calculate_gini(performances),
                "performance_spread": np.std(performances) / (np.mean(performances) + 1e-8)
            }

        return analysis

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or sum(values) == 0:
            return 0.0  # Perfect equality when all values are zero

        values = sorted(values)
        n = len(values)
        index = range(1, n + 1)
        return (2 * sum(index[i] * values[i] for i in range(n))) / (n * sum(values)) - (n + 1) / n


# Preset reward configurations for different research scenarios
class RewardPresets:
    """Predefined reward configurations for common scenarios."""

    @staticmethod
    def ant_colony_foraging() -> Dict[str, Any]:
        """Reward structure for ant-inspired foraging."""
        engineer = RewardEngineer("foraging")

        # Custom components for ant behavior
        engineer.add_component(
            InformationAsymmetryReward(
                info_value_fn=lambda aid, info: float(info['environment'].get('pheromone_followed', False)),
                weight=0.4
            )
        )

        engineer.add_component(
            CollectiveEfficiencyReward(
                individual_baseline_fn=lambda aid, ctx: ctx.environmental_state.get('random_foraging_rate', 0.1),
                efficiency_threshold=2.0,
                weight=0.6
            )
        )

        return {
            "engineer": engineer,
            "description": "Ant colony foraging with pheromone trail rewards"
        }

    @staticmethod
    def bee_waggle_dance() -> Dict[str, Any]:
        """Reward structure for bee-inspired resource communication."""
        engineer = RewardEngineer("resource_discovery")

        engineer.add_component(
            InformationAsymmetryReward(
                info_value_fn=lambda aid, info: info['environment'].get('found_distant_resource', 0.0),
                weight=0.5,
                warmup_steps=100
            )
        )

        engineer.add_component(
            EmergentSpecializationReward(
                n_roles=2,  # Scouts and foragers
                diversity_bonus=1.0,
                weight=0.5
            )
        )

        return {
            "engineer": engineer,
            "description": "Bee waggle dance for distant resource communication"
        }

    @staticmethod
    def neural_synchronization() -> Dict[str, Any]:
        """Reward structure for neural-inspired synchronization."""
        engineer = RewardEngineer("synchronization")

        engineer.add_component(
            TemporalCouplingReward(
                sync_window=10,
                sync_threshold=0.8,
                weight=0.7
            )
        )

        engineer.add_component(
            AdaptiveChallengeReward(
                base_difficulty=0.3,
                adaptation_rate=0.02,
                weight=0.3
            )
        )

        return {
            "engineer": engineer,
            "description": "Neural synchronization for collective computation"
        }


# Example usage
if __name__ == "__main__":
    print("🎯 Reward Engineering for Communication Emergence")
    print("=" * 50)

    # Example 1: Multi-principle foraging rewards
    print("\n1. Multi-Principle Foraging Environment")
    engineer = RewardEngineer("foraging")
    reward_structure = engineer.create_implicit_reward_structure("multi_principle")

    print(f"   Components: {len(reward_structure['components'])}")
    for component in reward_structure['components']:
        print(f"   - {component.get_principle().value}: weight={component.weight}")

    # Example 2: Curriculum learning
    print("\n2. Curriculum Learning Structure")
    engineer = RewardEngineer("general")
    curriculum = engineer.create_implicit_reward_structure("curriculum")

    print(f"   Stages: {len(curriculum['stages'])}")
    for i, stage in enumerate(curriculum['stages']):
        print(f"   Stage {i+1}: {len(stage['components'])} components, "
              f"duration={stage['duration']} steps")

    # Example 3: Test reward calculation
    print("\n3. Sample Reward Calculation")

    # Create test context
    test_context = RewardContext(
        agent_states={"agent_0": np.random.randn(10), "agent_1": np.random.randn(10)},
        agent_actions={"agent_0": np.array([0.8, 0.2]), "agent_1": np.array([0.7, 0.3])},
        agent_observations={"agent_0": np.random.randn(8), "agent_1": np.random.randn(8)},
        communication_signals={"agent_0": np.array([1, 0, 0, 0])},
        environmental_state={
            "task_completed": True,
            "collective_score": 10.0,
            "agent_0_found_food": True,
            "agent_0_sees_food": False
        },
        timestep=100
    )

    # Calculate rewards
    engineer = RewardEngineer("foraging")
    engineer.create_implicit_reward_structure("multi_principle")

    total_reward, components = engineer.calculate_reward("agent_0", test_context)

    print(f"   Total reward: {total_reward:.3f}")
    print("   Component breakdown:")
    for comp_name, value in components.items():
        print(f"     - {comp_name}: {value:.3f}")

    # Example 4: Analyze reward distribution
    print("\n4. Reward Distribution Analysis")

    # Simulate some reward history
    for step in range(50):
        test_context.timestep = step
        for agent in ["agent_0", "agent_1"]:
            engineer.calculate_reward(agent, test_context)

    analysis = engineer.analyze_reward_distribution()

    print("   Component contributions:")
    for comp, stats in analysis["component_contributions"].items():
        print(f"     - {comp}: mean={stats['mean']:.3f}, total={stats['total']:.1f}")

    print("\n✅ Reward engineering module ready for implicit communication encouragement!")
