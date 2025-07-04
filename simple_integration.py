# File: marlcomm/simple_integration_example.py
"""
Simple Integration Example - Quick Start Guide

This script shows the minimal code needed to:
1. Create a communication emergence environment
2. Apply implicit reward engineering
3. Train agents
4. Analyze emerged communication
5. Generate visualizations

Perfect for getting started quickly!
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from emergence_environments import create_emergence_environment
from reward_engineering import RewardEngineer, RewardContext
from communication_metrics import CommunicationAnalyzer, CommunicationEvent
from analysis_visualization import ExperimentData, CommunicationVisualizer, EmergenceAnalyzer


def simple_training_loop(env, reward_engineer, communication_analyzer, num_episodes=100):
    """
    Simplified training loop that demonstrates the integration.
    In practice, you'd use RLlib or another RL framework.
    """

    # Storage for analysis
    all_events = []
    episode_rewards = []
    metrics_history = {
        'mutual_information': [],
        'coordination_efficiency': [],
        'communication_frequency': []
    }

    print("🚀 Starting training...")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_events = []
        done = False
        step = 0

        while not done and step < 200:
            # Simple random policy (replace with your RL algorithm)
            actions = {}
            comm_signals = {}

            for agent in obs.keys():
                # Random action
                action = env.action_space(agent).sample()
                actions[agent] = action

                # Extract communication from action (last N dimensions)
                if hasattr(action, '__len__') and len(action) > 5:
                    comm_signals[agent] = action[5:]  # Communication part
                else:
                    comm_signals[agent] = np.array([0.0])

            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Create reward context for engineering
            context = RewardContext(
                agent_states={agent: obs[agent] for agent in actions},
                agent_actions=actions,
                agent_observations=obs,
                communication_signals=comm_signals,
                environmental_state={'step': step, 'episode': episode},
                timestep=episode * 200 + step
            )

            # Apply reward engineering
            engineered_rewards = {}
            for agent in actions:
                if agent in rewards:
                    total_reward, components = reward_engineer.calculate_reward(agent, context)
                    engineered_rewards[agent] = rewards[agent] + total_reward

            # Collect communication events
            for sender, signal in comm_signals.items():
                if np.any(signal != 0):  # Non-zero communication
                    event = CommunicationEvent(
                        timestep=context.timestep,
                        sender_id=sender,
                        receiver_ids=[a for a in actions if a != sender],
                        message=signal,
                        sender_state=obs[sender],
                        environmental_context={'episode': episode, 'step': step},
                        response_actions={r: actions[r] for r in actions if r != sender}
                    )
                    episode_events.append(event)
                    all_events.append(event)

            # Update for next step
            episode_reward += sum(engineered_rewards.values())
            obs = next_obs
            done = all(terminations.values()) or all(truncations.values())
            step += 1

        # Episode complete
        episode_rewards.append(episode_reward)

        # Analyze communication every 10 episodes
        if (episode + 1) % 10 == 0 and all_events:
            analysis = communication_analyzer.analyze_episode(all_events[-500:])

            # Extract metrics
            metrics = analysis['emergence_status']['current_metrics']
            metrics_history['mutual_information'].append(metrics.get('mutual_information', 0))
            metrics_history['coordination_efficiency'].append(metrics.get('coordination_efficiency', 0))
            metrics_history['communication_frequency'].append(len(episode_events) / max(step, 1))

            # Print progress
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Communication events: {len(episode_events)}")
            print(f"  Emergence score: {analysis['emergence_status']['emergence_score']:.3f}")

            if analysis['emergence_status']['emergence_detected']:
                print("  🎉 Communication has emerged!")

    return all_events, episode_rewards, metrics_history


def main():
    """Run a simple integrated experiment."""

    print("🌿 Simple Nature-Inspired MARL Communication Example")
    print("=" * 50)

    # 1. Create emergence environment
    print("\n1️⃣ Creating foraging environment...")
    env = create_emergence_environment(
        env_type="foraging",
        n_agents=3,
        grid_size=(20, 20),
        n_food_clusters=2
    )

    # 2. Setup reward engineering
    print("\n2️⃣ Setting up implicit rewards...")
    reward_engineer = RewardEngineer("foraging")
    reward_engineer.create_implicit_reward_structure("multi_principle")

    # 3. Initialize communication analyzer
    print("\n3️⃣ Initializing communication analysis...")
    communication_analyzer = CommunicationAnalyzer()

    # 4. Run training
    print("\n4️⃣ Training agents...")
    events, rewards, metrics = simple_training_loop(
        env, reward_engineer, communication_analyzer,
        num_episodes=50
    )

    # 5. Analyze results
    print("\n5️⃣ Analyzing communication patterns...")
    final_analysis = communication_analyzer.analyze_episode(events)

    # Print insights
    print("\n📊 Key Insights:")
    for insight in final_analysis['insights']:
        print(f"  {insight}")

    # 6. Create visualizations
    print("\n6️⃣ Creating visualizations...")

    # Create experiment data object
    exp_data = ExperimentData(
        name="Simple_Foraging_Demo",
        metrics_history=metrics,
        communication_events=events[-1000:],
        episode_rewards=rewards,
        agent_trajectories={},
        emergence_timestep=final_analysis['emergence_status']['emergence_timestep']
    )

    # Plot metrics evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')

    # Mutual information
    axes[0, 1].plot(metrics['mutual_information'])
    axes[0, 1].set_title('Mutual Information')
    axes[0, 1].set_xlabel('Analysis Point')
    axes[0, 1].set_ylabel('MI (bits)')

    # Coordination efficiency
    axes[1, 0].plot(metrics['coordination_efficiency'])
    axes[1, 0].set_title('Coordination Efficiency')
    axes[1, 0].set_xlabel('Analysis Point')
    axes[1, 0].set_ylabel('Efficiency')

    # Communication frequency
    axes[1, 1].plot(metrics['communication_frequency'])
    axes[1, 1].set_title('Communication Frequency')
    axes[1, 1].set_xlabel('Analysis Point')
    axes[1, 1].set_ylabel('Messages per Step')

    plt.tight_layout()
    plt.savefig('simple_integration_results.png')
    plt.show()

    # Create communication network visualization
    visualizer = CommunicationVisualizer()

    # Extract interaction matrix from events
    interaction_matrix = np.zeros((3, 3))  # 3 agents
    agent_names = [f"Agent_{i}" for i in range(3)]

    for event in events[-100:]:  # Last 100 events
        sender_idx = int(event.sender_id.split('_')[1])
        for receiver in event.receiver_ids:
            receiver_idx = int(receiver.split('_')[1])
            interaction_matrix[sender_idx, receiver_idx] += 1

    # Plot network
    network_fig = visualizer.plot_communication_network(
        interaction_matrix,
        agent_names,
        save_path='communication_network.png'
    )
    plt.show()

    # 7. Generate analysis report
    print("\n7️⃣ Generating analysis report...")
    report = communication_analyzer.generate_report(save_path='simple_analysis_report.txt')

    print("\n✅ Integration example complete!")
    print("📁 Output files:")
    print("   - simple_integration_results.png: Metrics evolution")
    print("   - communication_network.png: Final communication network")
    print("   - simple_analysis_report.txt: Detailed analysis report")

    # Print reward distribution analysis
    print("\n💰 Reward Engineering Analysis:")
    reward_analysis = reward_engineer.analyze_reward_distribution()

    for component, stats in reward_analysis['component_contributions'].items():
        print(f"   {component}: {stats['mean']:.3f} ± {stats['std']:.3f}")


if __name__ == "__main__":
    main()
