# File: marlcomm/communication_metrics.py
"""
Communication Metrics Module for Emergence Analysis

This module provides sophisticated metrics and analysis tools for measuring
when and how communication emerges in multi-agent systems. It includes
information-theoretic measures, behavioral analysis, and protocol detection.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import json


@dataclass
class CommunicationEvent:
    """Single communication event between agents."""
    timestep: int
    sender_id: str
    receiver_ids: List[str]
    message: np.ndarray
    sender_state: np.ndarray
    environmental_context: Dict[str, Any]
    response_actions: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ProtocolPattern:
    """Detected communication protocol pattern."""
    pattern_id: str
    frequency: int
    participants: List[str]
    message_sequence: List[np.ndarray]
    context_conditions: Dict[str, Any]
    success_rate: float
    stability_score: float


class MetricCalculator(ABC):
    """Base class for different metric calculators."""

    @abstractmethod
    def calculate(self, events: List[CommunicationEvent]) -> float:
        """Calculate metric from communication events."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass


class MutualInformationCalculator(MetricCalculator):
    """
    Calculate mutual information between messages and outcomes.
    High MI indicates messages carry meaningful information.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def calculate(self, events: List[CommunicationEvent]) -> float:
        if len(events) < 10:
            return 0.0

        # Extract messages and subsequent actions
        messages = []
        outcomes = []

        for i, event in enumerate(events[:-1]):
            if event.response_actions:
                # Flatten message
                msg_flat = event.message.flatten()
                messages.append(msg_flat)

                # Use response actions as outcomes
                responses = list(event.response_actions.values())
                if responses:
                    outcome = np.concatenate([r.flatten() for r in responses])
                    outcomes.append(outcome)

        if len(messages) < 5:
            return 0.0

        messages = np.array(messages)
        outcomes = np.array(outcomes)

        # Discretize for MI calculation
        msg_discrete = self._discretize(messages)
        out_discrete = self._discretize(outcomes)

        # Calculate mutual information
        mi = 0.0
        for msg_bin in range(self.n_bins):
            for out_bin in range(self.n_bins):
                p_msg = np.mean(msg_discrete == msg_bin)
                p_out = np.mean(out_discrete == out_bin)
                p_joint = np.mean((msg_discrete == msg_bin) & (out_discrete == out_bin))

                if p_joint > 0 and p_msg > 0 and p_out > 0:
                    mi += p_joint * np.log2(p_joint / (p_msg * p_out))

        return mi

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous data into bins."""
        if data.ndim > 1:
            # Use first principal component for multi-dimensional data
            data_centered = data - data.mean(axis=0)
            _, _, vt = np.linalg.svd(data_centered, full_matrices=False)
            data_1d = data_centered @ vt[0]
        else:
            data_1d = data.flatten()

        # Create bins
        bins = np.linspace(data_1d.min(), data_1d.max(), self.n_bins + 1)
        return np.digitize(data_1d, bins) - 1

    def get_name(self) -> str:
        return "mutual_information"


class ChannelCapacityEstimator(MetricCalculator):
    """
    Estimate the effective channel capacity being used.
    Higher capacity indicates richer communication.
    """

    def calculate(self, events: List[CommunicationEvent]) -> float:
        if len(events) < 10:
            return 0.0

        messages = np.array([e.message.flatten() for e in events])

        # Estimate entropy of message distribution
        # Use kernel density estimation for continuous messages
        kde_bandwidths = np.std(messages, axis=0) * 0.5
        kde_bandwidths[kde_bandwidths == 0] = 0.01

        # Sample from KDE to estimate entropy
        n_samples = min(1000, len(messages) * 10)
        samples = []
        for _ in range(n_samples):
            idx = np.random.randint(len(messages))
            sample = messages[idx] + np.random.randn(*messages[idx].shape) * kde_bandwidths
            samples.append(sample)

        samples = np.array(samples)

        # Estimate entropy using nearest neighbor method
        k = min(5, len(samples) // 10)
        distances = []
        for i in range(len(samples)):
            dists = np.linalg.norm(samples - samples[i], axis=1)
            dists[i] = np.inf  # Exclude self
            kth_dist = np.partition(dists, k)[k]
            distances.append(kth_dist)

        # Kozachenko-Leonenko estimator
        entropy = np.log(2 * k) + np.log(np.pi) / 2 + np.mean(np.log(distances))

        return max(0, entropy)

    def get_name(self) -> str:
        return "channel_capacity"


class CoordinationEfficiencyCalculator(MetricCalculator):
    """
    Measure how well communication leads to coordinated behavior.
    """

    def calculate(self, events: List[CommunicationEvent]) -> float:
        if len(events) < 5:
            return 0.0

        coordination_scores = []

        # Group events by timestep
        events_by_time = defaultdict(list)
        for event in events:
            events_by_time[event.timestep].append(event)

        # Analyze coordination at each timestep
        for timestep, step_events in events_by_time.items():
            if len(step_events) > 1:
                # Check if agents who communicated acted similarly
                communicators = set()
                actions = {}

                for event in step_events:
                    communicators.add(event.sender_id)
                    communicators.update(event.receiver_ids)

                    for agent_id, action in event.response_actions.items():
                        actions[agent_id] = action

                if len(actions) > 1:
                    # Calculate action similarity among communicating agents
                    action_list = list(actions.values())
                    similarities = []

                    for i in range(len(action_list)):
                        for j in range(i + 1, len(action_list)):
                            sim = 1 - np.linalg.norm(action_list[i] - action_list[j]) / (
                                np.linalg.norm(action_list[i]) + np.linalg.norm(action_list[j]) + 1e-8
                            )
                            similarities.append(sim)

                    if similarities:
                        coordination_scores.append(np.mean(similarities))

        return np.mean(coordination_scores) if coordination_scores else 0.0

    def get_name(self) -> str:
        return "coordination_efficiency"


class ProtocolStabilityAnalyzer(MetricCalculator):
    """
    Analyze the stability and consistency of communication protocols.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.protocol_history = deque(maxlen=window_size)

    def calculate(self, events: List[CommunicationEvent]) -> float:
        if len(events) < self.window_size:
            return 0.0

        # Group events into communication patterns
        patterns = self._extract_patterns(events[-self.window_size:])

        # Compare with historical patterns
        if hasattr(self, '_previous_patterns'):
            stability = self._calculate_pattern_similarity(patterns, self._previous_patterns)
        else:
            stability = 0.0

        self._previous_patterns = patterns

        return stability

    def _extract_patterns(self, events: List[CommunicationEvent]) -> List[Dict]:
        """Extract communication patterns from events."""
        patterns = []

        # Look for repeated message sequences
        for i in range(len(events) - 2):
            pattern = {
                'messages': [events[j].message for j in range(i, min(i + 3, len(events)))],
                'participants': set([events[j].sender_id for j in range(i, min(i + 3, len(events)))]),
                'context': events[i].environmental_context
            }
            patterns.append(pattern)

        return patterns

    def _calculate_pattern_similarity(self, patterns1: List[Dict], patterns2: List[Dict]) -> float:
        """Calculate similarity between two sets of patterns."""
        if not patterns1 or not patterns2:
            return 0.0

        similarities = []

        for p1 in patterns1[:10]:  # Compare top patterns
            best_sim = 0.0
            for p2 in patterns2[:10]:
                # Compare participants
                participant_sim = len(p1['participants'] & p2['participants']) / len(p1['participants'] | p2['participants'])

                # Compare messages
                msg_sim = 0.0
                for m1, m2 in zip(p1['messages'], p2['messages']):
                    if m1.shape == m2.shape:
                        msg_sim += 1 - np.linalg.norm(m1 - m2) / (np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-8)
                msg_sim /= len(p1['messages'])

                sim = 0.7 * msg_sim + 0.3 * participant_sim
                best_sim = max(best_sim, sim)

            similarities.append(best_sim)

        return np.mean(similarities)

    def get_name(self) -> str:
        return "protocol_stability"


class SemanticCoherenceAnalyzer(MetricCalculator):
    """
    Analyze whether similar situations produce similar messages (semantic coherence).
    """

    def calculate(self, events: List[CommunicationEvent]) -> float:
        if len(events) < 20:
            return 0.0

        # Group messages by similar environmental contexts
        context_messages = defaultdict(list)

        for event in events:
            # Create context signature
            context_key = self._get_context_signature(event.environmental_context)
            context_messages[context_key].append(event.message)

        # Calculate message consistency within similar contexts
        coherence_scores = []

        for context, messages in context_messages.items():
            if len(messages) > 2:
                # Calculate pairwise message similarities
                similarities = []
                for i in range(len(messages)):
                    for j in range(i + 1, len(messages)):
                        sim = 1 - np.linalg.norm(messages[i] - messages[j]) / (
                            np.linalg.norm(messages[i]) + np.linalg.norm(messages[j]) + 1e-8
                        )
                        similarities.append(sim)

                if similarities:
                    coherence_scores.append(np.mean(similarities))

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _get_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a hashable signature from context."""
        # Simple binning of numerical context values
        signature_parts = []
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, (int, float)):
                binned = int(value * 10) / 10  # Round to nearest 0.1
                signature_parts.append(f"{key}:{binned}")
            else:
                signature_parts.append(f"{key}:{value}")

        return "|".join(signature_parts)

    def get_name(self) -> str:
        return "semantic_coherence"


class CommunicationNetworkAnalyzer:
    """
    Analyze the communication network structure and dynamics.
    """

    def __init__(self):
        self.interaction_matrix = defaultdict(lambda: defaultdict(int))
        self.message_types = defaultdict(list)

    def update(self, event: CommunicationEvent):
        """Update network with new communication event."""
        for receiver in event.receiver_ids:
            self.interaction_matrix[event.sender_id][receiver] += 1

        # Cluster messages to identify types
        msg_flat = event.message.flatten()
        self.message_types[event.sender_id].append(msg_flat)

    def get_network_metrics(self) -> Dict[str, float]:
        """Calculate network-level metrics."""
        if not self.interaction_matrix:
            return {
                "network_density": 0.0,
                "centralization": 0.0,
                "clustering_coefficient": 0.0,
                "information_diversity": 0.0
            }

        # Convert to adjacency matrix
        agents = sorted(set(self.interaction_matrix.keys()) |
                       set(sum([list(v.keys()) for v in self.interaction_matrix.values()], [])))
        n = len(agents)
        adj_matrix = np.zeros((n, n))

        for i, sender in enumerate(agents):
            for j, receiver in enumerate(agents):
                adj_matrix[i, j] = self.interaction_matrix[sender].get(receiver, 0)

        # Network density
        possible_edges = n * (n - 1)
        actual_edges = np.sum(adj_matrix > 0)
        density = actual_edges / possible_edges if possible_edges > 0 else 0

        # Degree centralization
        out_degrees = np.sum(adj_matrix > 0, axis=1)
        max_degree = np.max(out_degrees) if len(out_degrees) > 0 else 0
        centralization = np.sum(max_degree - out_degrees) / ((n - 1) * (n - 2)) if n > 2 else 0

        # Clustering coefficient (transitivity)
        clustering = self._calculate_clustering(adj_matrix)

        # Information diversity (entropy of message types)
        diversity = self._calculate_message_diversity()

        return {
            "network_density": density,
            "centralization": centralization,
            "clustering_coefficient": clustering,
            "information_diversity": diversity
        }

    def _calculate_clustering(self, adj_matrix: np.ndarray) -> float:
        """Calculate clustering coefficient."""
        n = adj_matrix.shape[0]
        if n < 3:
            return 0.0

        triangles = 0
        triples = 0

        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            k = len(neighbors)
            if k >= 2:
                triples += k * (k - 1) / 2
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if adj_matrix[neighbors[j], neighbors[l]] > 0:
                            triangles += 1

        return triangles / triples if triples > 0 else 0.0

    def _calculate_message_diversity(self) -> float:
        """Calculate diversity of message types using entropy."""
        all_messages = []
        for agent_messages in self.message_types.values():
            all_messages.extend(agent_messages)

        if len(all_messages) < 10:
            return 0.0

        # Cluster messages to identify types
        from sklearn.cluster import KMeans
        n_clusters = min(5, len(all_messages) // 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(all_messages)

        # Calculate entropy
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(n_clusters)

        return entropy / max_entropy if max_entropy > 0 else 0.0


class EmergenceDetector:
    """
    Sophisticated emergence detection using multiple metrics and criteria.
    """

    def __init__(self, window_size: int = 100, emergence_threshold: float = 0.6):
        self.window_size = window_size
        self.emergence_threshold = emergence_threshold

        # Initialize metric calculators
        self.metrics = {
            "mutual_information": MutualInformationCalculator(),
            "channel_capacity": ChannelCapacityEstimator(),
            "coordination_efficiency": CoordinationEfficiencyCalculator(),
            "protocol_stability": ProtocolStabilityAnalyzer(),
            "semantic_coherence": SemanticCoherenceAnalyzer()
        }

        self.network_analyzer = CommunicationNetworkAnalyzer()
        self.event_history = deque(maxlen=window_size * 2)
        self.metric_history = defaultdict(list)
        self.emergence_detected = False
        self.emergence_timestep = None

    def update(self, events: List[CommunicationEvent]) -> Dict[str, Any]:
        """
        Update detector with new events and check for emergence.

        Returns:
            Dictionary with emergence status and current metrics
        """
        # Add events to history
        self.event_history.extend(events)

        # Update network analyzer
        for event in events:
            self.network_analyzer.update(event)

        # Calculate all metrics
        current_metrics = {}
        recent_events = list(self.event_history)[-self.window_size:]

        for metric_name, calculator in self.metrics.items():
            value = calculator.calculate(recent_events)
            current_metrics[metric_name] = value
            self.metric_history[metric_name].append(value)

        # Add network metrics
        network_metrics = self.network_analyzer.get_network_metrics()
        current_metrics.update(network_metrics)

        # Check for emergence using multiple criteria
        emergence_score = self._calculate_emergence_score(current_metrics)

        if not self.emergence_detected and emergence_score > self.emergence_threshold:
            self.emergence_detected = True
            self.emergence_timestep = len(self.event_history)

        return {
            "emergence_detected": self.emergence_detected,
            "emergence_score": emergence_score,
            "emergence_timestep": self.emergence_timestep,
            "current_metrics": current_metrics,
            "metric_trends": self._calculate_trends()
        }

    def _calculate_emergence_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall emergence score from individual metrics.
        Uses weighted combination with thresholds.
        """
        weights = {
            "mutual_information": 0.25,
            "coordination_efficiency": 0.20,
            "protocol_stability": 0.20,
            "semantic_coherence": 0.15,
            "network_density": 0.10,
            "information_diversity": 0.10
        }

        thresholds = {
            "mutual_information": 0.1,
            "coordination_efficiency": 0.5,
            "protocol_stability": 0.6,
            "semantic_coherence": 0.4,
            "network_density": 0.2,
            "information_diversity": 0.3
        }

        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize by threshold
                normalized_value = metrics[metric] / thresholds.get(metric, 1.0)
                score += weight * min(1.0, normalized_value)

        return score

    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate trend direction for each metric."""
        trends = {}

        for metric, history in self.metric_history.items():
            if len(history) > 10:
                recent = np.mean(history[-10:])
                older = np.mean(history[-20:-10])

                if recent > older * 1.1:
                    trends[metric] = "increasing"
                elif recent < older * 0.9:
                    trends[metric] = "decreasing"
                else:
                    trends[metric] = "stable"
            else:
                trends[metric] = "insufficient_data"

        return trends

    def visualize_emergence(self, save_path: Optional[str] = None):
        """Create visualization of emergence metrics over time."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = list(self.metrics.keys()) + ["network_density"]

        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                ax = axes[idx]

                if metric in self.metric_history:
                    history = self.metric_history[metric]
                elif metric == "network_density":
                    # Get network density history
                    history = []
                    for i in range(0, len(self.event_history), 10):
                        events_slice = list(self.event_history)[max(0, i-50):i]
                        if events_slice:
                            for e in events_slice:
                                self.network_analyzer.update(e)
                            metrics = self.network_analyzer.get_network_metrics()
                            history.append(metrics["network_density"])
                else:
                    continue

                if history:
                    ax.plot(history, linewidth=2)
                    ax.set_title(metric.replace("_", " ").title(), fontsize=12)
                    ax.set_xlabel("Time Steps")
                    ax.set_ylabel("Value")
                    ax.grid(True, alpha=0.3)

                    # Mark emergence point
                    if self.emergence_timestep:
                        ax.axvline(x=self.emergence_timestep, color='red',
                                 linestyle='--', label='Emergence Detected')
                        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()

        plt.close()


class CommunicationAnalyzer:
    """
    Main class for comprehensive communication analysis.
    Integrates all metrics and provides high-level insights.
    """

    def __init__(self):
        self.emergence_detector = EmergenceDetector()
        self.protocol_patterns = []
        self.analysis_results = {}

    def analyze_episode(self, events: List[CommunicationEvent]) -> Dict[str, Any]:
        """
        Perform complete analysis of communication in an episode.

        Returns:
            Comprehensive analysis results
        """
        # Update emergence detector
        emergence_status = self.emergence_detector.update(events)

        # Extract communication protocols
        protocols = self._extract_protocols(events)

        # Analyze message semantics
        semantic_analysis = self._analyze_semantics(events)

        # Generate insights
        insights = self._generate_insights(emergence_status, protocols, semantic_analysis)

        results = {
            "emergence_status": emergence_status,
            "detected_protocols": protocols,
            "semantic_analysis": semantic_analysis,
            "insights": insights,
            "summary_statistics": self._calculate_summary_stats(events)
        }

        self.analysis_results = results
        return results

    def _extract_protocols(self, events: List[CommunicationEvent]) -> List[ProtocolPattern]:
        """Extract recurring communication protocols."""
        protocols = []

        # Look for repeated sequences
        sequence_length = 3
        sequences = defaultdict(list)

        for i in range(len(events) - sequence_length):
            # Create sequence signature
            seq_messages = [events[j].message for j in range(i, i + sequence_length)]
            seq_participants = [events[j].sender_id for j in range(i, i + sequence_length)]

            # Use JSON for safe serialization
            seq_hash = json.dumps(seq_participants)
            sequences[seq_hash].append({
                'messages': seq_messages,
                'participants': seq_participants,
                'start_idx': i,
                'success': self._evaluate_sequence_success(events[i:i+sequence_length])
            })

        # Identify stable protocols
        for seq_hash, instances in sequences.items():
            if len(instances) > 3:  # Repeated at least 3 times
                success_rate = np.mean([inst['success'] for inst in instances])

                # Parse participants safely from the hash
                # seq_hash is a string representation of participants list
                # Use json to safely parse it
                try:
                    participants = json.loads(seq_hash)
                except (json.JSONDecodeError, TypeError):
                    # Fallback: extract from instances
                    participants = list(set(sum([inst.get('participants', [])
                                               for inst in instances], [])))

                protocol = ProtocolPattern(
                    pattern_id=f"protocol_{len(protocols)}",
                    frequency=len(instances),
                    participants=participants,
                    message_sequence=instances[0]['messages'],
                    context_conditions={},
                    success_rate=success_rate,
                    stability_score=self._calculate_protocol_stability(instances)
                )
                protocols.append(protocol)

        return protocols

    def _evaluate_sequence_success(self, sequence_events: List[CommunicationEvent]) -> float:
        """Evaluate if a communication sequence led to successful coordination."""
        # Check if receivers responded appropriately
        if not sequence_events:
            return 0.0

        response_count = sum(len(e.response_actions) for e in sequence_events)
        expected_responses = sum(len(e.receiver_ids) for e in sequence_events)

        return response_count / max(expected_responses, 1)

    def _calculate_protocol_stability(self, instances: List[Dict]) -> float:
        """Calculate how stable a protocol is across instances."""
        if len(instances) < 2:
            return 0.0

        # Compare message similarity across instances
        similarities = []

        for i in range(len(instances) - 1):
            msgs1 = instances[i]['messages']
            msgs2 = instances[i + 1]['messages']

            sim = 0.0
            for m1, m2 in zip(msgs1, msgs2):
                if m1.shape == m2.shape:
                    sim += 1 - np.linalg.norm(m1 - m2) / (np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-8)
            sim /= len(msgs1)
            similarities.append(sim)

        return np.mean(similarities)

    def _analyze_semantics(self, events: List[CommunicationEvent]) -> Dict[str, Any]:
        """Analyze the semantic content of messages."""
        if not events:
            return {"message_clusters": 0, "context_correlation": 0.0}

        # Collect all messages with contexts
        messages = []
        contexts = []

        for event in events:
            messages.append(event.message.flatten())
            # Extract numerical features from context
            context_features = []
            for key, value in sorted(event.environmental_context.items()):
                if isinstance(value, (int, float)):
                    context_features.append(value)
            contexts.append(context_features)

        messages = np.array(messages)

        # Cluster messages to find semantic groups
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=3)
        message_labels = clustering.fit_predict(messages)

        n_clusters = len(set(message_labels)) - (1 if -1 in message_labels else 0)

        # Calculate context-message correlation
        correlation = 0.0
        if contexts and len(contexts[0]) > 0:
            contexts = np.array(contexts)
            # Use first principal component of messages
            msg_pca = messages - messages.mean(axis=0)
            if msg_pca.shape[1] > 0:
                u, s, vt = np.linalg.svd(msg_pca, full_matrices=False)
                msg_pc1 = msg_pca @ vt[0]

                # Correlate with each context dimension
                correlations = []
                for i in range(contexts.shape[1]):
                    if np.std(contexts[:, i]) > 0:
                        corr = np.abs(np.corrcoef(msg_pc1, contexts[:, i])[0, 1])
                        correlations.append(corr)

                correlation = np.mean(correlations) if correlations else 0.0

        return {
            "message_clusters": n_clusters,
            "context_correlation": correlation,
            "semantic_diversity": n_clusters / max(len(events) / 10, 1)  # Normalized
        }

    def _generate_insights(self, emergence_status: Dict, protocols: List[ProtocolPattern],
                          semantic_analysis: Dict) -> List[str]:
        """Generate human-readable insights from analysis."""
        insights = []

        # Emergence insights
        if emergence_status["emergence_detected"]:
            insights.append(f"✅ Communication emerged at timestep {emergence_status['emergence_timestep']}")
            insights.append(f"   Current emergence score: {emergence_status['emergence_score']:.2f}")
        else:
            insights.append(f"⏳ Communication emerging... Score: {emergence_status['emergence_score']:.2f}")

        # Metric insights
        metrics = emergence_status["current_metrics"]
        if metrics.get("mutual_information", 0) > 0.2:
            insights.append("📊 High mutual information - messages carry meaningful content")

        if metrics.get("coordination_efficiency", 0) > 0.7:
            insights.append("🤝 Strong coordination - agents acting cohesively")

        # Protocol insights
        if protocols:
            insights.append(f"🔄 Detected {len(protocols)} stable communication protocols")
            best_protocol = max(protocols, key=lambda p: p.success_rate)
            insights.append(f"   Best protocol success rate: {best_protocol.success_rate:.1%}")

        # Semantic insights
        if semantic_analysis["message_clusters"] > 3:
            insights.append(f"💬 Rich vocabulary - {semantic_analysis['message_clusters']} distinct message types")

        if semantic_analysis["context_correlation"] > 0.5:
            insights.append("🎯 Context-aware - messages correlate with environmental state")

        # Network insights
        if metrics.get("network_density", 0) > 0.3:
            insights.append("🌐 Dense communication network formed")

        # Trend insights
        trends = emergence_status.get("metric_trends", {})
        improving = [m for m, t in trends.items() if t == "increasing"]
        if improving:
            insights.append(f"📈 Improving: {', '.join(improving)}")

        return insights

    def _calculate_summary_stats(self, events: List[CommunicationEvent]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not events:
            return {}

        total_messages = len(events)
        unique_senders = len(set(e.sender_id for e in events))

        # Message rate over time
        if events:
            time_span = events[-1].timestep - events[0].timestep + 1
            message_rate = total_messages / time_span
        else:
            message_rate = 0

        # Average receivers per message
        avg_receivers = np.mean([len(e.receiver_ids) for e in events])

        return {
            "total_messages": total_messages,
            "unique_senders": unique_senders,
            "message_rate": message_rate,
            "avg_receivers_per_message": avg_receivers,
            "total_timesteps": events[-1].timestep if events else 0
        }

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive analysis report."""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_episode first."

        report = []
        report.append("=" * 60)
        report.append("COMMUNICATION EMERGENCE ANALYSIS REPORT")
        report.append("=" * 60)

        # Summary
        report.append("\n## SUMMARY")
        stats = self.analysis_results.get("summary_statistics", {})
        report.append(f"Total Messages: {stats.get('total_messages', 0)}")
        report.append(f"Message Rate: {stats.get('message_rate', 0):.2f} per timestep")
        report.append(f"Unique Communicators: {stats.get('unique_senders', 0)}")

        # Emergence Status
        report.append("\n## EMERGENCE STATUS")
        emergence = self.analysis_results["emergence_status"]
        if emergence["emergence_detected"]:
            report.append(f"✅ EMERGENCE DETECTED at timestep {emergence['emergence_timestep']}")
        else:
            report.append("⏳ EMERGENCE IN PROGRESS")
        report.append(f"Emergence Score: {emergence['emergence_score']:.2f}")

        # Metrics
        report.append("\n## KEY METRICS")
        metrics = emergence["current_metrics"]
        for metric, value in sorted(metrics.items()):
            report.append(f"{metric:.<30} {value:.3f}")

        # Protocols
        report.append("\n## DETECTED PROTOCOLS")
        protocols = self.analysis_results.get("detected_protocols", [])
        if protocols:
            for i, protocol in enumerate(protocols[:5]):  # Top 5
                report.append(f"\nProtocol {i+1}:")
                report.append(f"  Frequency: {protocol.frequency}")
                report.append(f"  Success Rate: {protocol.success_rate:.1%}")
                report.append(f"  Stability: {protocol.stability_score:.2f}")
                report.append(f"  Participants: {protocol.participants}")
        else:
            report.append("No stable protocols detected yet")

        # Insights
        report.append("\n## KEY INSIGHTS")
        for insight in self.analysis_results.get("insights", []):
            report.append(insight)

        # Join report
        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


# Example usage
if __name__ == "__main__":
    print("🔬 Communication Metrics Module")
    print("=" * 50)

    # Create analyzer
    analyzer = CommunicationAnalyzer()

    # Simulate some communication events
    print("\nSimulating communication events...")
    events = []

    for t in range(200):
        # Create realistic communication pattern
        if t > 50:  # Communication emerges after exploration
            sender = f"agent_{t % 3}"
            receivers = [f"agent_{(t + 1) % 3}", f"agent_{(t + 2) % 3}"]

            # Message content evolves over time
            if t < 100:
                message = np.random.randn(4) * 0.5
            else:
                # More structured messages
                message = np.array([
                    np.sin(t * 0.1),
                    np.cos(t * 0.1),
                    1.0 if t % 10 < 5 else -1.0,
                    0.5
                ])

            event = CommunicationEvent(
                timestep=t,
                sender_id=sender,
                receiver_ids=receivers,
                message=message,
                sender_state=np.random.randn(10),
                environmental_context={
                    "resource_nearby": t % 20 < 10,
                    "danger_level": 0.0 if t < 150 else 0.5
                },
                response_actions={
                    receivers[0]: message * 0.8 + np.random.randn(4) * 0.1,
                    receivers[1]: message * 0.7 + np.random.randn(4) * 0.2
                }
            )
            events.append(event)

    # Analyze
    print(f"\nAnalyzing {len(events)} communication events...")
    results = analyzer.analyze_episode(events)

    # Generate report
    print("\n" + analyzer.generate_report())

    # Visualize
    print("\nGenerating emergence visualization...")
    analyzer.emergence_detector.visualize_emergence("emergence_metrics.png")
    print("✅ Visualization saved to emergence_metrics.png")
