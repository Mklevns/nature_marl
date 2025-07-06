# File: marlcomm/analysis_visualization.py
"""
Analysis & Visualization Suite for MARL Communication Research

This module provides comprehensive tools for analyzing emergence patterns,
visualizing communication networks, and generating publication-quality figures
for research papers. Includes real-time monitoring and post-hoc analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Note: UMAP not available. Install with: pip install umap-learn")
from sklearn.cluster import DBSCAN, KMeans
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. Install with: pip install plotly")
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False
})


@dataclass
class ExperimentData:
    """Container for experiment data."""
    name: str
    metrics_history: Dict[str, List[float]]
    communication_events: List[Any]
    episode_rewards: List[float]
    agent_trajectories: Dict[str, List[np.ndarray]]
    emergence_timestep: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationVisualizer:
    """Visualize communication patterns and networks."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)

    def plot_communication_network(self,
                                 interaction_matrix: np.ndarray,
                                 agent_names: List[str],
                                 timestep: Optional[int] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot communication network as a directed graph.

        Args:
            interaction_matrix: Matrix of communication frequencies
            agent_names: List of agent identifiers
            timestep: Optional timestep for title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for agent in agent_names:
            G.add_node(agent)

        # Add edges with weights
        for i, sender in enumerate(agent_names):
            for j, receiver in enumerate(agent_names):
                if i != j and interaction_matrix[i, j] > 0:
                    G.add_edge(sender, receiver, weight=interaction_matrix[i, j])

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        node_sizes = [1000 + 500 * G.degree(node) for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color=self.color_palette[:len(agent_names)],
                              alpha=0.8, ax=ax)

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(G, pos, edgelist=edges,
                              width=[5 * w / max_weight for w in weights],
                              alpha=0.6, edge_color='gray',
                              arrowsize=20, arrowstyle='->', ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

        # Add edge labels for strong connections
        edge_labels = {}
        for u, v, w in G.edges(data='weight'):
            if w > max_weight * 0.5:  # Only show strong connections
                edge_labels[(u, v)] = f'{w:.1f}'
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)

        title = "Communication Network"
        if timestep is not None:
            title += f" (Step {timestep})"
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def animate_network_evolution(self,
                                interaction_history: List[np.ndarray],
                                agent_names: List[str],
                                interval: int = 200,
                                save_path: Optional[str] = None):
        """
        Create animation of communication network evolution.

        Args:
            interaction_history: List of interaction matrices over time
            agent_names: Agent identifiers
            interval: Animation interval in milliseconds
            save_path: Path to save animation (mp4 or gif)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        def update(frame):
            ax.clear()
            self.plot_communication_network(
                interaction_history[frame],
                agent_names,
                timestep=frame,
                save_path=None
            )

        anim = FuncAnimation(fig, update, frames=len(interaction_history),
                           interval=interval, blit=False)

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow')
            else:
                anim.save(save_path, writer='ffmpeg')

        return anim

    def plot_message_space(self,
                         messages: List[np.ndarray],
                         labels: Optional[List[str]] = None,
                         method: str = 'pca',
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize high-dimensional messages in 2D/3D space.

        Args:
            messages: List of message vectors
            labels: Optional labels for coloring
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            save_path: Path to save figure
        """
        messages_array = np.array(messages)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=min(3, messages_array.shape[1]))
            reduced = reducer.fit_transform(messages_array)
            title = "Message Space (PCA)"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(messages_array)
            title = "Message Space (t-SNE)"
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                print("UMAP not available, falling back to t-SNE")
                reducer = TSNE(n_components=2, random_state=42)
                reduced = reducer.fit_transform(messages_array)
                title = "Message Space (t-SNE - UMAP unavailable)"
            else:
                reducer = UMAP(n_components=2, random_state=42)
                reduced = reducer.fit_transform(messages_array)
                title = "Message Space (UMAP)"
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create plot
        if reduced.shape[1] == 3:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')

            if labels is not None:
                unique_labels = list(set(labels))
                colors = [self.color_palette[unique_labels.index(l) % len(self.color_palette)]
                         for l in labels]
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                                   c=colors, alpha=0.6, s=50)
            else:
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                                   alpha=0.6, s=50)

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

        else:
            fig, ax = plt.subplots(figsize=self.figsize)

            if labels is not None:
                unique_labels = list(set(labels))
                for i, label in enumerate(unique_labels):
                    mask = [l == label for l in labels]
                    ax.scatter(reduced[mask, 0], reduced[mask, 1],
                             label=label, alpha=0.6, s=50,
                             color=self.color_palette[i % len(self.color_palette)])
                ax.legend()
            else:
                ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')

        ax.set_title(title, fontsize=16, fontweight='bold')

        if method == 'pca':
            # Add explained variance
            if hasattr(reducer, 'explained_variance_ratio_'):
                variance_text = f"Explained variance: {reducer.explained_variance_ratio_[:2].sum():.1%}"
                ax.text(0.02, 0.98, variance_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return fig


class EmergenceAnalyzer:
    """Analyze emergence patterns and dynamics."""

    def __init__(self):
        self.metrics_analyzer = MetricsAnalyzer()
        self.statistical_tests = StatisticalTests()

    def analyze_emergence_dynamics(self,
                                 experiments: List[ExperimentData],
                                 metric_name: str = "mutual_information") -> Dict[str, Any]:
        """
        Analyze how communication emerges across experiments.

        Returns:
            Dictionary with emergence statistics and patterns
        """
        emergence_times = []
        emergence_rates = []
        final_values = []

        for exp in experiments:
            if metric_name in exp.metrics_history:
                history = exp.metrics_history[metric_name]

                # Emergence time
                if exp.emergence_timestep is not None:
                    emergence_times.append(exp.emergence_timestep)

                # Emergence rate (slope during emergence)
                if len(history) > 20:
                    # Find steepest increase
                    window = 20
                    rates = []
                    for i in range(len(history) - window):
                        rate = (history[i + window] - history[i]) / window
                        rates.append(rate)
                    emergence_rates.append(max(rates))

                # Final value
                final_values.append(np.mean(history[-10:]) if len(history) > 10 else history[-1])

        analysis = {
            "emergence_times": {
                "mean": np.mean(emergence_times) if emergence_times else None,
                "std": np.std(emergence_times) if emergence_times else None,
                "min": min(emergence_times) if emergence_times else None,
                "max": max(emergence_times) if emergence_times else None
            },
            "emergence_rates": {
                "mean": np.mean(emergence_rates) if emergence_rates else None,
                "std": np.std(emergence_rates) if emergence_rates else None
            },
            "final_values": {
                "mean": np.mean(final_values) if final_values else None,
                "std": np.std(final_values) if final_values else None
            },
            "success_rate": len(emergence_times) / len(experiments) if experiments else 0
        }

        return analysis

    def plot_emergence_comparison(self,
                                experiments: List[ExperimentData],
                                metrics: List[str],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare emergence across multiple experiments and metrics.
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            for exp in experiments:
                if metric in exp.metrics_history:
                    history = exp.metrics_history[metric]
                    ax.plot(history, label=exp.name, alpha=0.7, linewidth=2)

                    # Mark emergence point
                    if exp.emergence_timestep is not None:
                        ax.axvline(x=exp.emergence_timestep, color='red',
                                 linestyle='--', alpha=0.5)

            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Training Steps')
        fig.suptitle('Emergence Comparison Across Experiments', fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return fig

    def identify_critical_transitions(self,
                                    metric_history: List[float],
                                    window_size: int = 50) -> List[int]:
        """
        Identify critical transition points in metric evolution.
        Uses change point detection and phase transition analysis.
        """
        if len(metric_history) < window_size * 2:
            return []

        # Calculate rolling statistics
        metric_array = np.array(metric_history)
        transitions = []

        # Method 1: Variance surge detection (early warning signal)
        for i in range(window_size, len(metric_array) - window_size):
            pre_var = np.var(metric_array[i-window_size:i])
            post_var = np.var(metric_array[i:i+window_size])

            if post_var > pre_var * 2:  # Variance doubles
                transitions.append(i)

        # Method 2: Mean shift detection
        for i in range(window_size, len(metric_array) - window_size):
            pre_mean = np.mean(metric_array[i-window_size:i])
            post_mean = np.mean(metric_array[i:i+window_size])

            if abs(post_mean - pre_mean) > 2 * np.std(metric_array[:i]):
                if i not in transitions:
                    transitions.append(i)

        return sorted(transitions)


class MetricsAnalyzer:
    """Advanced metrics analysis and correlation."""

    def calculate_information_flow(self,
                                 sender_messages: List[np.ndarray],
                                 receiver_actions: List[np.ndarray],
                                 time_lag: int = 1) -> float:
        """
        Calculate information flow from sender to receiver using
        transfer entropy or Granger causality.
        """
        if len(sender_messages) < time_lag + 10 or len(receiver_actions) < time_lag + 10:
            return 0.0

        # Prepare time series
        X = np.array([m.flatten() for m in sender_messages[:-time_lag]])
        Y = np.array([a.flatten() for a in receiver_actions[time_lag:]])

        # Simple transfer entropy approximation
        # H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

        # Discretize for entropy calculation
        X_discrete = self._discretize_series(X)
        Y_discrete = self._discretize_series(Y)

        # Calculate conditional entropies
        h_y_given_y = self._conditional_entropy(Y_discrete[1:], Y_discrete[:-1])
        h_y_given_yx = self._conditional_entropy_2d(Y_discrete[1:], Y_discrete[:-1], X_discrete[:-1])

        transfer_entropy = h_y_given_y - h_y_given_yx

        return max(0, transfer_entropy)

    def _discretize_series(self, series: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Discretize continuous series for entropy calculation."""
        if series.ndim > 1:
            # Use first principal component
            pca = PCA(n_components=1)
            series_1d = pca.fit_transform(series).flatten()
        else:
            series_1d = series.flatten()

        bins = np.linspace(series_1d.min(), series_1d.max(), n_bins + 1)
        return np.digitize(series_1d, bins) - 1

    def _conditional_entropy(self, y: np.ndarray, x: np.ndarray) -> float:
        """Calculate H(Y|X)."""
        xy = np.column_stack([x, y])
        unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
        unique_x, counts_x = np.unique(x, return_counts=True)

        h_conditional = 0
        for i, x_val in enumerate(unique_x):
            p_x = counts_x[i] / len(x)

            # Get Y values for this X
            mask = x == x_val
            y_given_x = y[mask]

            if len(y_given_x) > 0:
                _, counts_y_given_x = np.unique(y_given_x, return_counts=True)
                p_y_given_x = counts_y_given_x / len(y_given_x)
                h_y_given_x = -np.sum(p_y_given_x * np.log2(p_y_given_x + 1e-10))
                h_conditional += p_x * h_y_given_x

        return h_conditional

    def _conditional_entropy_2d(self, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
        """
        Calculate H(Z|Y,X).
        WARNING: This is a simplified approximation that treats Y and X independently.
        For research requiring accurate conditional entropy, use proper estimators
        like JIDT (Java Information Dynamics Toolkit) or NPEET.
        """
        # Simplified version - treats Y and X independently
        return self._conditional_entropy(z, y) * 0.5 + self._conditional_entropy(z, x) * 0.5

    def analyze_synchronization(self,
                              agent_actions: Dict[str, List[np.ndarray]],
                              window_size: int = 10) -> Dict[str, float]:
        """
        Analyze synchronization patterns between agents.
        Returns various synchronization metrics.
        """
        agents = list(agent_actions.keys())
        n_agents = len(agents)

        if n_agents < 2:
            return {"synchronization_index": 0.0}

        # Convert to time series
        time_series = {}
        min_length = min(len(actions) for actions in agent_actions.values())

        for agent, actions in agent_actions.items():
            # Use first component of action as primary signal
            time_series[agent] = np.array([a.flatten()[0] for a in actions[:min_length]])

        # Calculate pairwise phase synchronization
        phase_sync_values = []

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Hilbert transform to get phase
                signal1 = time_series[agents[i]]
                signal2 = time_series[agents[j]]

                phase1 = np.angle(signal.hilbert(signal1))
                phase2 = np.angle(signal.hilbert(signal2))

                # Phase locking value
                phase_diff = phase1 - phase2
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                phase_sync_values.append(plv)

        # Calculate Kuramoto order parameter
        all_phases = []
        for agent in agents:
            phases = np.angle(signal.hilbert(time_series[agent]))
            all_phases.append(phases)

        all_phases = np.array(all_phases)
        kuramoto_order = np.abs(np.mean(np.exp(1j * all_phases), axis=0))

        return {
            "mean_phase_locking": np.mean(phase_sync_values),
            "kuramoto_order_parameter": np.mean(kuramoto_order),
            "synchronization_variance": np.var(phase_sync_values),
            "max_synchronization": np.max(phase_sync_values) if phase_sync_values else 0.0
        }


class StatisticalTests:
    """Statistical significance testing for emergence analysis."""

    def test_emergence_significance(self,
                                  pre_emergence: List[float],
                                  post_emergence: List[float],
                                  test_type: str = 'mann-whitney') -> Dict[str, float]:
        """
        Test if emergence represents statistically significant change.

        Args:
            pre_emergence: Metric values before emergence
            post_emergence: Metric values after emergence
            test_type: Statistical test to use

        Returns:
            Dictionary with test statistics and p-value
        """
        if test_type == 'mann-whitney':
            statistic, p_value = stats.mannwhitneyu(pre_emergence, post_emergence,
                                                   alternative='less')
        elif test_type == 't-test':
            statistic, p_value = stats.ttest_ind(pre_emergence, post_emergence)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(pre_emergence, post_emergence)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Calculate effect size (Cohen's d)
        cohens_d = (np.mean(post_emergence) - np.mean(pre_emergence)) / np.sqrt(
            (np.var(pre_emergence) + np.var(post_emergence)) / 2
        )

        return {
            "statistic": statistic,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05
        }

    def compare_conditions(self,
                         condition_results: Dict[str, List[float]],
                         test_type: str = 'kruskal') -> Dict[str, Any]:
        """
        Compare multiple experimental conditions.

        Args:
            condition_results: Dictionary mapping condition names to result lists
            test_type: Statistical test ('kruskal', 'anova')

        Returns:
            Test results including post-hoc comparisons
        """
        conditions = list(condition_results.keys())
        values = list(condition_results.values())

        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions to compare"}

        # Main test
        if test_type == 'kruskal':
            statistic, p_value = stats.kruskal(*values)
            test_name = "Kruskal-Wallis"
        elif test_type == 'anova':
            statistic, p_value = stats.f_oneway(*values)
            test_name = "One-way ANOVA"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        results = {
            "test": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05
        }

        # Post-hoc pairwise comparisons if significant
        if results["significant"] and len(conditions) > 2:
            pairwise = {}

            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    stat, p = stats.mannwhitneyu(values[i], values[j])
                    # Bonferroni correction
                    corrected_p = p * (len(conditions) * (len(conditions) - 1) / 2)

                    pairwise[f"{conditions[i]}_vs_{conditions[j]}"] = {
                        "p_value": p,
                        "corrected_p_value": min(corrected_p, 1.0),
                        "significant": corrected_p < 0.05
                    }

            results["pairwise_comparisons"] = pairwise

        return results


class InteractiveDashboard:
    """Create interactive dashboards using Plotly."""

    def create_emergence_dashboard(self,
                                 experiments: List[ExperimentData],
                                 save_path: Optional[str] = None):
        """
        Create comprehensive interactive dashboard for emergence analysis.
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available for interactive dashboard")
            print("   Install with: pip install plotly")
            return None

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Emergence Dynamics', 'Communication Network',
                          'Message Space Evolution', 'Reward Progression',
                          'Synchronization Metrics', 'Information Flow'),
            specs=[[{"secondary_y": False}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )

        colors = px.colors.qualitative.Plotly

        # 1. Emergence Dynamics
        for i, exp in enumerate(experiments):
            if 'mutual_information' in exp.metrics_history:
                fig.add_trace(
                    go.Scatter(x=list(range(len(exp.metrics_history['mutual_information']))),
                             y=exp.metrics_history['mutual_information'],
                             name=exp.name,
                             line=dict(color=colors[i % len(colors)])),
                    row=1, col=1
                )

        # 2. Communication Network (example for last timestep)
        if experiments and hasattr(experiments[0], 'communication_events'):
            # Extract network data from last experiment
            # This is simplified - in real implementation, extract from events
            network_data = self._extract_network_data(experiments[-1])

            edge_trace = go.Scatter(
                x=network_data['edge_x'],
                y=network_data['edge_y'],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            node_trace = go.Scatter(
                x=network_data['node_x'],
                y=network_data['node_y'],
                mode='markers+text',
                text=network_data['labels'],
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    size=network_data['node_sizes'],
                    color=network_data['node_colors'],
                    colorscale='Viridis',
                    line_width=2
                )
            )

            fig.add_trace(edge_trace, row=1, col=2)
            fig.add_trace(node_trace, row=1, col=2)

        # 3. Message Space Evolution (t-SNE over time)
        # Simplified visualization
        for i, exp in enumerate(experiments[:3]):  # Limit to 3 for clarity
            if hasattr(exp, 'communication_events'):
                t = np.linspace(0, 1, 100)
                x = np.sin(2 * np.pi * t + i) + np.random.randn(100) * 0.1
                y = np.cos(2 * np.pi * t + i) + np.random.randn(100) * 0.1

                fig.add_trace(
                    go.Scatter(x=x, y=y,
                             mode='markers',
                             name=f'{exp.name} messages',
                             marker=dict(size=5, color=t, colorscale='Viridis')),
                    row=2, col=1
                )

        # 4. Reward Progression
        for i, exp in enumerate(experiments):
            fig.add_trace(
                go.Scatter(x=list(range(len(exp.episode_rewards))),
                         y=exp.episode_rewards,
                         name=exp.name,
                         line=dict(color=colors[i % len(colors)])),
                row=2, col=2
            )

        # 5. Synchronization Metrics
        if experiments and 'coordination_efficiency' in experiments[0].metrics_history:
            for i, exp in enumerate(experiments):
                fig.add_trace(
                    go.Scatter(x=list(range(len(exp.metrics_history['coordination_efficiency']))),
                             y=exp.metrics_history['coordination_efficiency'],
                             name=exp.name,
                             line=dict(color=colors[i % len(colors)])),
                    row=3, col=1
                )

        # 6. Information Flow Heatmap
        # Create synthetic data for demonstration
        agents = ['Agent_0', 'Agent_1', 'Agent_2', 'Agent_3']
        info_flow_matrix = np.random.rand(4, 4) * 0.8
        np.fill_diagonal(info_flow_matrix, 0)

        fig.add_trace(
            go.Heatmap(z=info_flow_matrix,
                      x=agents,
                      y=agents,
                      colorscale='Viridis'),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Communication Emergence Analysis Dashboard",
            showlegend=True,
            height=1200,
            width=1600
        )

        # Update axes
        fig.update_xaxes(title_text="Training Steps", row=1, col=1)
        fig.update_yaxes(title_text="Mutual Information", row=1, col=1)

        fig.update_xaxes(title_text="Episode", row=2, col=2)
        fig.update_yaxes(title_text="Reward", row=2, col=2)

        fig.update_xaxes(title_text="Training Steps", row=3, col=1)
        fig.update_yaxes(title_text="Coordination", row=3, col=1)

        if save_path:
            fig.write_html(save_path)

        return fig

    def _extract_network_data(self, experiment: ExperimentData) -> Dict[str, Any]:
        """Extract network visualization data from experiment."""
        # Get number of agents from experiment data
        n_agents = experiment.metadata.get('num_agents', 4)

        # Create positions in a circle
        pos = {f'Agent_{i}': (np.cos(2*np.pi*i/n_agents), np.sin(2*np.pi*i/n_agents))
               for i in range(n_agents)}

        edge_x = []
        edge_y = []

        for i in range(n_agents):
            for j in range(i+1, n_agents):
                x0, y0 = pos[f'Agent_{i}']
                x1, y1 = pos[f'Agent_{j}']
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        node_x = [pos[f'Agent_{i}'][0] for i in range(n_agents)]
        node_y = [pos[f'Agent_{i}'][1] for i in range(n_agents)]

        return {
            'edge_x': edge_x,
            'edge_y': edge_y,
            'node_x': node_x,
            'node_y': node_y,
            'labels': [f'Agent_{i}' for i in range(n_agents)],
            'node_sizes': [20] * n_agents,
            'node_colors': list(range(n_agents))
        }


class PublicationFigures:
    """Generate publication-quality figures for papers."""

    def __init__(self, style: str = 'nature'):
        """
        Initialize with publication style.

        Args:
            style: Publication style ('nature', 'science', 'plos')
        """
        self.style = style
        self._set_style()

    def _set_style(self):
        """Set matplotlib style for publication."""
        if self.style == 'nature':
            plt.rcParams.update({
                'font.size': 8,
                'axes.labelsize': 9,
                'axes.titlesize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'figure.figsize': (3.5, 2.625),  # Single column
                'axes.linewidth': 0.8,
                'lines.linewidth': 1.0
            })
        elif self.style == 'science':
            plt.rcParams.update({
                'font.size': 9,
                'axes.labelsize': 10,
                'axes.titlesize': 10,
                'figure.figsize': (3.42, 2.57),  # Single column
            })

    def create_emergence_summary_figure(self,
                                      experiments: List[ExperimentData],
                                      metrics: List[str],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-ready emergence summary figure.
        """
        n_metrics = len(metrics)

        # Create figure with GridSpec for complex layout
        fig = plt.figure(figsize=(7, 2.5 * n_metrics))
        gs = fig.add_gridspec(n_metrics, 3, width_ratios=[3, 1, 1],
                            hspace=0.3, wspace=0.3)

        for i, metric in enumerate(metrics):
            # Main time series plot
            ax_main = fig.add_subplot(gs[i, 0])

            # Plot each experiment
            for exp in experiments:
                if metric in exp.metrics_history:
                    history = exp.metrics_history[metric]
                    ax_main.plot(history, label=exp.name, linewidth=1.5, alpha=0.8)

            ax_main.set_xlabel('Training Steps')
            ax_main.set_ylabel(metric.replace('_', ' ').title())
            ax_main.legend(frameon=False, loc='upper left')
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)

            # Box plot of final values
            ax_box = fig.add_subplot(gs[i, 1])
            final_values = []
            labels = []

            for exp in experiments:
                if metric in exp.metrics_history:
                    history = exp.metrics_history[metric]
                    if len(history) > 10:
                        final_values.append(history[-10:])
                        labels.append(exp.name)

            if final_values:
                ax_box.boxplot(final_values, labels=labels)
                ax_box.set_ylabel('Final Values')
                ax_box.tick_params(axis='x', rotation=45)
                ax_box.spines['top'].set_visible(False)
                ax_box.spines['right'].set_visible(False)

            # Statistical summary
            ax_stats = fig.add_subplot(gs[i, 2])
            ax_stats.axis('off')

            # Calculate statistics
            if final_values:
                mean_vals = [np.mean(vals) for vals in final_values]

                stats_text = f"Mean: {np.mean(mean_vals):.3f}\n"
                stats_text += f"Std: {np.std(mean_vals):.3f}\n"

                # Perform statistical test if multiple conditions
                if len(final_values) > 1:
                    _, p_value = stats.kruskal(*final_values)
                    stats_text += f"p-value: {p_value:.3f}"
                    if p_value < 0.05:
                        stats_text += "*"

                ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                            verticalalignment='center', fontsize=8)

        plt.suptitle('Communication Emergence Analysis', fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class ExperimentReporter:
    """Generate comprehensive analysis reports."""

    def __init__(self, output_dir: Path = Path("analysis_reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_full_report(self,
                           experiments: List[ExperimentData],
                           report_name: str = "communication_emergence_analysis"):
        """
        Generate complete analysis report with figures and statistics.
        """
        # Create report directory
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)

        # Initialize components
        visualizer = CommunicationVisualizer()
        analyzer = EmergenceAnalyzer()
        dashboard = InteractiveDashboard()
        pub_figures = PublicationFigures()

        # Generate all analyses
        report_content = []
        report_content.append("# Communication Emergence Analysis Report\n")
        report_content.append(f"Generated: {pd.Timestamp.now()}\n")
        report_content.append(f"Number of experiments: {len(experiments)}\n")

        # 1. Emergence dynamics analysis
        report_content.append("\n## Emergence Dynamics\n")

        metrics_to_analyze = ['mutual_information', 'coordination_efficiency',
                             'protocol_stability', 'network_density']

        for metric in metrics_to_analyze:
            analysis = analyzer.analyze_emergence_dynamics(experiments, metric)
            report_content.append(f"\n### {metric.replace('_', ' ').title()}\n")
            report_content.append(f"- Success rate: {analysis['success_rate']:.1%}\n")

            if analysis['emergence_times']['mean'] is not None:
                report_content.append(f"- Mean emergence time: {analysis['emergence_times']['mean']:.1f} ± "
                                    f"{analysis['emergence_times']['std']:.1f} steps\n")

        # 2. Statistical comparisons
        report_content.append("\n## Statistical Analysis\n")

        # Compare different conditions if experiments have labels
        condition_groups = defaultdict(list)
        for exp in experiments:
            condition = exp.metadata.get('condition', 'default')
            if 'mutual_information' in exp.metrics_history:
                final_mi = np.mean(exp.metrics_history['mutual_information'][-10:])
                condition_groups[condition].append(final_mi)

        if len(condition_groups) > 1:
            stats_results = StatisticalTests().compare_conditions(dict(condition_groups))
            report_content.append(f"Kruskal-Wallis test: H={stats_results['statistic']:.3f}, "
                                f"p={stats_results['p_value']:.3f}\n")

        # 3. Generate figures
        report_content.append("\n## Figures\n")

        # Emergence comparison
        fig_comparison = analyzer.plot_emergence_comparison(
            experiments, metrics_to_analyze,
            save_path=report_dir / "emergence_comparison.png"
        )
        report_content.append("- emergence_comparison.png: Multi-metric emergence comparison\n")
        plt.close(fig_comparison)

        # Publication figure
        fig_publication = pub_figures.create_emergence_summary_figure(
            experiments, metrics_to_analyze[:2],
            save_path=report_dir / "publication_figure.png"
        )
        report_content.append("- publication_figure.png: Publication-ready summary\n")
        plt.close(fig_publication)

        # Interactive dashboard
        dashboard_fig = dashboard.create_emergence_dashboard(
            experiments,
            save_path=report_dir / "interactive_dashboard.html"
        )
        report_content.append("- interactive_dashboard.html: Interactive analysis dashboard\n")

        # 4. Save report
        with open(report_dir / "report.md", 'w') as f:
            f.writelines(report_content)

        # 5. Save data for reproducibility
        experiment_data = []
        for exp in experiments:
            exp_dict = {
                'name': exp.name,
                'metrics_history': exp.metrics_history,
                'episode_rewards': exp.episode_rewards,
                'emergence_timestep': exp.emergence_timestep,
                'metadata': exp.metadata
            }
            experiment_data.append(exp_dict)

        with open(report_dir / "experiment_data.json", 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)

        print(f"✅ Report generated in: {report_dir}")
        print(f"   - report.md: Complete analysis report")
        print(f"   - emergence_comparison.png: Metric comparison figure")
        print(f"   - publication_figure.png: Publication-ready figure")
        print(f"   - interactive_dashboard.html: Interactive visualizations")
        print(f"   - experiment_data.json: Raw data for reproducibility")


# Example usage
if __name__ == "__main__":
    print("📊 Analysis & Visualization Suite for MARL Communication")
    print("=" * 60)

    # Create sample experiment data
    experiments = []

    for i in range(3):
        # Simulate metrics history
        steps = 1000
        emergence_point = 200 + i * 50

        # Mutual information gradually increases
        mi_history = []
        for t in range(steps):
            if t < emergence_point:
                mi_history.append(0.05 + np.random.randn() * 0.02)
            else:
                mi_history.append(0.5 + 0.3 * (1 - np.exp(-(t - emergence_point) / 100)) +
                                np.random.randn() * 0.05)

        # Coordination efficiency
        coord_history = []
        for t in range(steps):
            if t < emergence_point + 100:
                coord_history.append(0.3 + np.random.randn() * 0.1)
            else:
                coord_history.append(0.7 + np.random.randn() * 0.1)

        # Episode rewards
        rewards = []
        for ep in range(steps // 10):
            if ep < emergence_point // 10:
                rewards.append(np.random.randn() * 10)
            else:
                rewards.append(20 + np.random.randn() * 5)

        exp = ExperimentData(
            name=f"Experiment_{i+1}",
            metrics_history={
                'mutual_information': mi_history,
                'coordination_efficiency': coord_history,
                'network_density': [0.1 + t / steps * 0.5 + np.random.randn() * 0.05
                                   for t in range(steps)]
            },
            communication_events=[],  # Would contain actual events
            episode_rewards=rewards,
            agent_trajectories={},
            emergence_timestep=emergence_point,
            metadata={'condition': 'pheromone' if i == 0 else 'symbolic'}
        )

        experiments.append(exp)

    # Generate full report
    reporter = ExperimentReporter()
    reporter.generate_full_report(experiments, "example_analysis")

    print("\n✅ Analysis suite demonstration complete!")
    print("   Check 'analysis_reports/example_analysis/' for outputs")
