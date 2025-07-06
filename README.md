# 🌿 Nature-Inspired Emergent Communication in Multi-Agent RL

This project is a research framework for investigating how complex, efficient, and biologically-inspired communication protocols can emerge in multi-agent systems. It uses **Ray RLlib 2.9.0**, **PettingZoo**, and a modern PyTorch-based stack to build, train, and analyze sophisticated agents that learn to coordinate and communicate from scratch.

The core of this framework is to create environmental pressures that necessitate cooperation, and then provide agents with the architectural tools to develop their own communication protocols to solve the task. This is inspired by how communication evolved in nature—as a survival advantage.

-----

## Core Features

  * **🧬 Biologically-Inspired Environments**: Custom multi-agent environments designed to simulate natural challenges that drive communication. These are compatible with the PettingZoo API and include scenarios like resource scarcity (Foraging) and threat avoidance (Predator-Prey).
  * **🧠 Advanced Communication Models**: The agent's "brain" is a custom `RLModule` called `NatureInspiredCommModule`. It goes beyond simple message passing, incorporating:
      * **Multi-Round Attention**: For targeted, contextual communication between agents, inspired by collective decision-making.
      * **Pheromone-like Signaling**: Agents generate bounded, continuous message vectors analogous to chemical trails.
      * **Neural Plasticity**: A GRU-based memory system allows agents to integrate experiences and communication over time.
  * **🔬 Deep Emergence Analytics**: The framework includes a `RealEmergenceTrackingCallbacks` class that provides deep insights into the learned protocol. It moves beyond simple task rewards to measure the quality of the communication itself, tracking metrics such as:
      * Mutual Information
      * Coordination Efficiency
      * Protocol Stability
      * Semantic Coherence
  * **⚡ Hardware-Optimized Training**: The framework automatically detects system hardware (GPU, CPU, RAM) and configures the training pipeline to maximize throughput, enabling faster research cycles on high-end systems.

-----

## Core Concept

This project is built on foundational concepts in multi-agent reinforcement learning.

### Emergent Communication (EC)

Instead of manually designing a rigid, hard-coded communication protocol, this framework allows agents to develop their own. A shared protocol is learned concurrently with the agents' policies, guided only by a team-based reward signal. The "meaning" of a message emerges from its utility in solving the task.

This approach is a powerful alternative to manual design, which is often brittle and sub-optimal, especially as the number of agents and task complexity grows.

### Foundational Architectures

The model design is inspired by several seminal works in the field:

  * **RIAL (Reinforced Inter-Agent Learning)**: The basic idea of treating communication as a learnable action that is part of an agent's policy.
  * **DIAL (Differentiable Inter-Agent Learning)**: The key insight that a continuous communication channel during training allows gradients to flow from the "listener" back to the "speaker". This provides a much richer learning signal than task reward alone and is a core principle behind our `RLModule` design.
  * **CommNet & Attention**: The idea of aggregating information from multiple agents is central. Our framework modernizes this by using multi-head attention (similar to TarMAC and MAGIC) to allow agents to learn *who* to listen to, rather than just averaging all messages.

### Centralized Training, Decentralized Execution (CTDE)

This is a standard and powerful paradigm in MARL that we fully embrace. During **training**, we can use centralized information—like sharing gradients through a differentiable channel or using a centralized value function—to make learning more stable and efficient. During **execution**, however, the agents act purely based on their local observations and the messages they receive, without a central controller.

-----

## 🚀 Project Structure

This repository is organized into modular components, each with a specific responsibility:

| File/Module                                          | Role                                                                                                                                                                             |
| -----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main_trainer.py`                                    | **Main Entry Point.** The unified script to run all experiments. Parses arguments and orchestrates the entire training pipeline.                                                 |
| `marlcomm/models/rl_module.py`                       | **The Agent's Brain.** Defines the `NatureInspiredCommModule`, which contains the neural network architecture for perception, memory, and communication.                         |
| `marlcomm/enviroments/emergence_environments.py`     | **The Simulation Worlds.** Implements various `PettingZoo`-style multi-agent environments designed to create pressures for communication to emerge.                              |
| `marlcomm/utils/wrappers.py`                         | **Metrics & Reward Injection.** A crucial `PettingZoo` wrapper that calculates implicit rewards and captures `CommunicationEvent` objects for analysis by the callbacks.         |
| `marlcomm/utils/callbacks.py`                        | **The Analysis Engine.** An RLlib `DefaultCallbacks` class that calculates and logs all custom emergence and hardware performance metrics during training.                       |
| `marlcomm/config/training_config.py`                 | **Hardware-Aware Config Factory.** Detects system hardware and generates an optimized `PPOConfig` to maximize training performance.                                              |
| `marlcomm/utils/metrics.py`                          | **Metrics Definitions.** Defines the `CommunicationAnalyzer` and the data structures used to quantify communication patterns like mutual information and protocol stability.     |
| `marlcomm/config/ray_config.py`                      | **Centralized Ray Config.** Provides helper functions for initializing Ray and creating baseline PPO configurations according to the new API stack.                              |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 🛠️ Setup and Installation

This project is built on a specific, verified stack to ensure compatibility and reproducibility. It is highly recommended to use a Python virtual environment.

**1. Create and Activate a Virtual Environment**

Using `venv`:

```bash
python3 -m venv marl-env
source marl-env/bin/activate
```

Or using `conda`:

```bash
conda create -n marl-env python=3.10
conda activate marl-env
```

**2. Install Dependencies**

Save the following to a `requirements.txt` file. These versions are verified to work together seamlessly.

```text
# Verified requirements for Ray 2.9.x MARL
ray[rllib]==2.9.0
torch>=2.0.0
numpy==1.24.3
gymnasium==0.28.1
pettingzoo>=1.24.0
supersuit==3.8.0

# System & Hardware Monitoring
psutil
gputil

# Visualization
matplotlib
tensorboard>=2.11.0
```

Install the packages using pip:

```bash
pip install -r requirements.txt
```

-----

## ▶️ How to Run an Experiment

All experiments are run through the unified `main_trainer.py` script. You can configure the environment, agent models, and training parameters using command-line arguments.

Use the `--help` flag to see all available options:

```bash
python main_trainer.py --help
```

### Example Commands

**1. Run a default foraging experiment with 8 agents on auto-detected hardware:**

```bash
python main_trainer.py --env foraging --agents 8 --iterations 100
```

**2. Force GPU training for a predator-prey scenario with more agents and longer training:**

```bash
python main_trainer.py --hardware gpu --env predator_prey --agents 12 --iterations 500
```

**3. Run a quick test on CPU with a smaller environment to ensure everything works:**

```bash
python main_trainer.py --hardware cpu --agents 3 --grid-size 15 15 --iterations 10
```

-----

## 📊 Analyzing Results

Training results, logs, and model checkpoints are saved to the `marl_results/` directory by default (or the directory specified with `--output-dir`).

### TensorBoard Visualization

RLlib integrates with TensorBoard to provide real-time visualization of training metrics. To launch it, run the following command in your terminal:

```bash
tensorboard --logdir marl_results/
```

Navigate to `http://localhost:6006` in your browser. You will find standard RL metrics (`episode_reward_mean`, etc.) as well as all the **custom emergence metrics** logged by our callback system, such as:

  * `custom/emergence_score_mean`
  * `custom/mutual_information_mean`
  * `custom/coordination_efficiency_mean`
  * `custom/protocol_stability_mean`
  * `custom/communication_frequency_mean`
  * `custom/gpu_utilization_percent` (if applicable)

Analyzing these custom metrics is key to understanding *how* the agents are solving the task, not just *how well*.
