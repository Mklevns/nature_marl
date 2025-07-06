# Migration Guide: Hardware Optimization → Unified Training Config

## 🔄 Quick Migration Reference

### **Before (Separate Modules)**
```python
# Old imports
from hardware_optimization import HardwareOptimizer, OptimizedTrainingPipeline, OptimizedCallbacks
from training_config import TrainingConfigFactory, NatureCommCallbacks, print_hardware_summary

# Old usage pattern
optimizer = HardwareOptimizer()
factory = TrainingConfigFactory(optimizer.profile)
pipeline = OptimizedTrainingPipeline("experiment")
config = factory.create_config(obs_space, act_space)
```

### **After (Unified Module)**
```python
# New imports
from training_config import NatureInspiredTrainingFactory, print_nature_inspired_hardware_summary

# New usage pattern
factory = NatureInspiredTrainingFactory()
config = factory.create_adaptive_config(obs_space, act_space)
pipeline = factory.create_optimized_training_pipeline("experiment")
```

## 📝 Class Mapping Reference

| Old Class | New Class | Notes |
|-----------|-----------|-------|
| `HardwareOptimizer` | `NatureInspiredHardwareOptimizer` | Enhanced with bio-inspired parameters |
| `TrainingConfigFactory` | `NatureInspiredTrainingFactory` | Integrated hardware optimization |
| `OptimizedTrainingPipeline` | Methods in `NatureInspiredTrainingFactory` | Consolidated as factory methods |
| `OptimizedCallbacks` + `NatureCommCallbacks` | `EmergentBehaviorTracker` | Merged callback system |
| `HardwareProfile` + `HardwareInfo` | `HardwareProfile` | Unified hardware representation |

## 🔧 Method Migration

### **Hardware Detection & Optimization**
```python
# Before
optimizer = HardwareOptimizer()
profile = optimizer.profile
ppo_config = optimizer.get_optimized_ppo_config(env, env_config, rl_module_spec)

# After
factory = NatureInspiredTrainingFactory()
profile = factory.hardware_profile
ppo_config = factory.create_adaptive_config(obs_space, act_space, env_config)
```

### **Ray Initialization**
```python
# Before
ray_config = optimizer.optimize_ray_init()
ray.init(**ray_config)

# After
ray_config = factory.get_ray_initialization_config()
ray.init(**ray_config)
```

### **Training Pipeline**
```python
# Before
pipeline = OptimizedTrainingPipeline("experiment")
train_config = pipeline.create_optimized_training_config(env_name, env_config, rl_module_spec)
results = pipeline.run_optimized_training(train_config)

# After
pipeline_config = factory.create_optimized_training_pipeline("experiment")
ray.init(**pipeline_config["ray_config"])
# Use pipeline_config with tune.run() or Algorithm.train()
```

### **Callbacks**
```python
# Before
config.callbacks(OptimizedCallbacks)
# or
config.callbacks(NatureCommCallbacks)

# After
config.callbacks(EmergentBehaviorTracker)  # Combines both systems
```

## 🌿 Enhanced Bio-Inspired Features

### **New Terminology**
| Technical Term | Bio-Inspired Term | Meaning |
|----------------|-------------------|---------|
| `batch_size` | `neural_batch_size` | Neural burst processing patterns |
| `num_workers` | `worker_colonies` | Ant colony-inspired worker distribution |
| `learning_rate` | `neural_plasticity_rate` | Synaptic adaptation speed |
| `entropy_coeff` | `adaptive_entropy_coeff` | Exploration drive like curiosity |
| `memory_size` | `memory_consolidation_capacity` | Hippocampal-like memory system |

### **New Bio-Inspired Metrics**
```python
# Available in EmergentBehaviorTracker
episode.custom_metrics["pheromone_entropy"]                    # Communication diversity
episode.custom_metrics["swarm_coordination_index"]           # Collective coordination
episode.custom_metrics["neural_plasticity_rate"]             # Learning adaptation
episode.custom_metrics["memory_consolidation_strength"]      # Experience integration
episode.custom_metrics["emergent_strategy_complexity"]       # Behavioral sophistication
```

## ⚙️ Configuration Method Updates

### **Basic Configuration**
```python
# Before
factory = TrainingConfigFactory()
config = factory.create_config(obs_space, act_space, force_mode="gpu")

# After
factory = NatureInspiredTrainingFactory()
config = factory.create_adaptive_config(obs_space, act_space, force_mode="gpu")
```

### **Specific Hardware Targeting**
```python
# Before
config = factory.create_gpu_config(obs_space, act_space)
config = factory.create_cpu_config(obs_space, act_space)

# After
config = factory.create_gpu_optimized_config(obs_space, act_space, env_config)
config = factory.create_cpu_optimized_config(obs_space, act_space, env_config)
```

### **Tune Configuration**
```python
# Before
tune_config = factory.get_tune_config(num_iterations=100)

# After
tune_config = factory.get_tune_config(num_iterations=100, experiment_name="nature_marl")
```

## 🔍 Hardware Summary Function

```python
# Before
from hardware_optimization import HardwareOptimizer
from training_config import print_hardware_summary

optimizer = HardwareOptimizer()  # Prints automatically
print_hardware_summary()

# After
from training_config import print_nature_inspired_hardware_summary

print_nature_inspired_hardware_summary()
```

## 📊 Accessing Hardware Information

```python
# Before
optimizer = HardwareOptimizer()
gpu_available = optimizer.profile.gpu_available
batch_size = optimizer.profile.optimal_batch_size
workers = optimizer.profile.optimal_num_workers

# After
factory = NatureInspiredTrainingFactory()
gpu_available = factory.hardware_profile.gpu_available
batch_size = factory.hardware_profile.optimal_batch_size
workers = factory.hardware_profile.optimal_num_workers

# New bio-inspired parameters
comm_channels = factory.hardware_profile.communication_channels
plasticity_rate = factory.hardware_profile.recommended_learning_rate
memory_consolidation = factory.hardware_profile.memory_consolidation_frequency
```

## 🚨 Breaking Changes to Address

### **1. Import Updates Required**
- Replace all `hardware_optimization` imports with `training_config`
- Update class names to new bio-inspired versions
- Remove duplicate imports (both modules imported the same dependencies)

### **2. Method Signature Changes**
```python
# Before
get_optimized_ppo_config(env: str, env_config: Dict, rl_module_spec: Any)

# After
create_adaptive_config(obs_space, act_space, env_config: Dict = None, force_mode: str = None)
```

### **3. Callback Integration**
- `OptimizedCallbacks` and `NatureCommCallbacks` are now `EmergentBehaviorTracker`
- New metrics available with bio-inspired names
- Enhanced tracking of emergent behaviors

### **4. Configuration Object Changes**
- `HardwareProfile` now includes bio-inspired parameters
- Additional fields for neural plasticity and communication
- Enhanced memory management parameters

## ✅ Migration Checklist

- [ ] Update imports from separate modules to unified `training_config`
- [ ] Replace `HardwareOptimizer()` with `NatureInspiredTrainingFactory()`
- [ ] Update method calls to new unified API
- [ ] Replace callback classes with `EmergentBehaviorTracker`
- [ ] Update any direct hardware profile access
- [ ] Test bio-inspired parameter access
- [ ] Verify Ray initialization configuration
- [ ] Update experiment configuration calls
- [ ] Test hardware summary function
- [ ] Update any custom extensions or subclasses

## 🧪 Testing Your Migration

```python
# Test script to verify migration
from training_config import NatureInspiredTrainingFactory, print_nature_inspired_hardware_summary

# Test 1: Hardware detection
print_nature_inspired_hardware_summary()

# Test 2: Factory creation
factory = NatureInspiredTrainingFactory()
assert factory.hardware_profile is not None

# Test 3: Configuration creation
import gymnasium as gym
obs_space = gym.spaces.Box(0, 1, (4,))
act_space = gym.spaces.Discrete(2)
config = factory.create_adaptive_config(obs_space, act_space)
assert config is not None

# Test 4: Ray configuration
ray_config = factory.get_ray_initialization_config()
assert "num_cpus" in ray_config

print("✅ Migration successful!")
```

## 📞 Support & Troubleshooting

### **Common Issues**
1. **Import Errors**: Ensure you've updated all imports to the unified module
2. **Method Not Found**: Check the method mapping table above
3. **Missing Parameters**: New bio-inspired parameters are optional with sensible defaults
4. **Callback Errors**: Replace old callback classes with `EmergentBehaviorTracker`

### **Performance Verification**
The unified module maintains all performance optimizations while adding bio-inspired enhancements. If you experience performance regression, verify:
- Hardware detection is working correctly
- Ray configuration matches your expectations
- Batch sizes and worker counts are appropriate
- GPU utilization (if applicable) remains optimal

Your migration to the unified bio-inspired training configuration system will provide enhanced performance monitoring and improved maintainability!
