# nature_marl
Nature-Inspired Multi-Agent Reinforcement Learning
# 🌿 Nature-Inspired Multi-Agent Reinforcement Learning

A modular multi-agent reinforcement learning system implementing biological communication patterns inspired by nature. Agents learn to coordinate using pheromone-like signals, neural plasticity, and adaptive memory mechanisms.

## 🚀 Quick Start

```bash
# Test system compatibility
python debug_utils.py

# Run training with auto-detected hardware
python main_trainer.py

# Force GPU training (if available)
python main_trainer.py --gpu

# Force CPU training
python main_trainer.py --cpu

# Run longer training session
python main_trainer.py --iterations 50
```

## 📁 Modular Architecture

The system is split into focused modules for easy debugging and development:

```
nature_marl/
├── main_trainer.py       # Main entry point and orchestration
├── rl_module.py         # Custom neural network with bio-inspired communication
├── environment.py       # Multi-agent environment setup and testing
├── training_config.py   # Hardware-aware training configurations
├── debug_utils.py       # Testing and debugging utilities
└── README.md           # This file
```

### 🧠 `rl_module.py` - Nature-Inspired Neural Networks
- **Pheromone Trails**: Bounded communication signals between agents
- **Neural Plasticity**: GRU-based memory that adapts over time
- **Adaptive Behavior**: Combines observations with communication history
- **New API Stack**: Fully compatible with Ray RLlib 2.40+ requirements

### 🌍 `environment.py` - Multi-Agent Environment
- **PettingZoo Integration**: Uses MPE simple_spread scenario
- **Preprocessing**: Applies SuperSuit wrappers for compatibility
- **Testing**: Comprehensive environment validation
- **Configuration**: Flexible environment parameter tuning

### ⚙️ `training_config.py` - Hardware-Aware Training
- **Auto-Detection**: Automatically detects GPU/CPU capabilities
- **Resource Optimization**: Tailored configs for different hardware
- **Memory Safety**: Conservative settings to prevent OOM errors
- **New API Stack**: Uses latest Ray RLlib parameter names

### 🔧 `debug_utils.py` - Testing & Debugging
- **Comprehensive Tests**: Validates all system components
- **Hardware Verification**: Tests PyTorch, CUDA, and Ray functionality
- **Modular Testing**: Test individual components in isolation
- **Diagnostics**: Detailed error reporting and troubleshooting

## 🛠️ Installation

### Requirements
```bash
pip install ray[rllib] torch gymnasium mpe2 supersuit numpy
```

### Optional (for detailed hardware info)
```bash
pip install psutil
```

### Verify Installation
```bash
python debug_utils.py --test all
```

## 🧬 Biological Inspiration

The system implements several nature-inspired communication mechanisms:

### 🐜 Pheromone Trails
- **Chemical Signaling**: Agents produce bounded communication signals
- **Information Persistence**: Signals maintain information over time
- **Spatial Coordination**: Enables landmark coverage coordination

### 🧠 Neural Plasticity  
- **Experience Integration**: GRU-based memory adapts to agent interactions
- **Learning Transfer**: Past experiences inform current decisions
- **Dynamic Adaptation**: Memory patterns evolve during training

### 🐝 Information Encoding
- **Signal Processing**: Agents encode/decode communication messages
- **Adaptive Behavior**: Decision-making integrates multiple information sources
- **Collective Intelligence**: Emergent coordination from individual behaviors

## 📊 Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **Python**: 3.8+
- **PyTorch**: 1.11+

### Recommended for GPU Training
- **GPU**: 8+ GB VRAM (RTX 3070+, RTX 4060+)
- **CPU**: 8+ cores (Ryzen 7+, Intel i7+)
- **RAM**: 16+ GB
- **CUDA**: 11.7+

### Your System (Auto-Detected)
```bash
python training_config.py  # Shows detected hardware
```

## 🎯 Usage Examples

### Basic Training
```bash
# Auto-detect best training mode
python main_trainer.py
```

### Advanced Options
```bash
# Test system without training
python main_trainer.py --test-only

# Custom training length
python main_trainer.py --iterations 100

# Force specific hardware mode
python main_trainer.py --gpu --iterations 30
```

### Debugging
```bash
# Test specific components
python debug_utils.py --test env      # Test environment only
python debug_utils.py --test gpu      # Test GPU functionality
python debug_utils.py --test module   # Test neural network
```

## 🔬 Research Applications

This system is designed for research in:

- **Multi-Agent Communication**: Study emergent communication protocols
- **Bio-Inspired AI**: Explore nature-inspired coordination mechanisms  
- **Swarm Intelligence**: Investigate collective decision-making
- **Adaptive Systems**: Research neural plasticity in RL
- **Coordination Learning**: Analyze multi-agent cooperation strategies

## 📈 Performance Optimization

### GPU Training (RTX 4070 Example)
- **Expected Speed**: 10-20x faster than CPU
- **Memory Usage**: ~4-8GB VRAM, ~8-16GB RAM
- **Batch Size**: 1024-2048 samples per learner
- **Workers**: 4-6 environment runners

### CPU Training (Ryzen 9 3900X Example)  
- **Workers**: 8-12 environment runners
- **Memory Usage**: ~4-8GB RAM
- **Batch Size**: 512-1024 samples per learner
- **Performance**: Stable training, longer iterations

## 🐛 Troubleshooting

### Common Issues

#### GPU Out of Memory
```bash
# Try CPU mode
python main_trainer.py --cpu

# Or reduce batch sizes in training_config.py
```

#### Environment Errors
```bash
# Test environment separately
python debug_utils.py --test env

# Check MPE2 installation
pip install --upgrade mpe2
```

#### Ray Errors
```bash
# Test Ray installation
python debug_utils.py --test all

# Update Ray
pip install --upgrade ray[rllib]
```

#### Import Errors
```bash
# Verify all dependencies
python debug_utils.py --test deps

# Reinstall missing packages
pip install -r requirements.txt
```

## 📝 Customization

### Modify Communication Patterns
Edit `rl_module.py`:
```python
# Adjust communication dimensions
self.comm_dim = 16  # Increase signal complexity
self.memory_dim = 32  # Expand memory capacity
```

### Change Environment Parameters
Edit `environment.py`:
```python
# Modify environment settings
config = EnvironmentConfig(
    num_agents=5,        # More agents
    max_cycles=50,       # Longer episodes
    local_ratio=0.3      # Different reward structure
)
```

### Adjust Training Settings
Edit `training_config.py`:
```python
# Modify training parameters
.training(
    train_batch_size_per_learner=2048,  # Larger batches
    num_epochs=15,                      # More training epochs
    lr=5e-4,                           # Different learning rate
)
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** with `python debug_utils.py`
4. **Submit** a pull request

## 📚 References

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [PettingZoo Multi-Agent Environments](https://pettingzoo.farama.org/)
- [Multi-Agent Particle Environments](https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe)
- [Nature-Inspired Computation](https://en.wikipedia.org/wiki/Bio-inspired_computing)

## 📄 License

MIT License - Feel free to use and modify for research and educational purposes.

---

**Happy Training! 🎉**

*May your agents swarm with the wisdom of nature and the power of deep learning.*
