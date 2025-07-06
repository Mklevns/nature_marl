"""Bio-inspired models for multi-agent RL."""

from .rl_module import (
    ProductionUnifiedBioInspiredRLModule,
    ProductionPheromoneAttentionNetwork,
    ProductionNeuralPlasticityMemory,
    ProductionMultiActionHead,
    create_production_bio_module_spec
)

# Create convenient aliases
BioInspiredRLModule = ProductionUnifiedBioInspiredRLModule
PheromoneAttentionNetwork = ProductionPheromoneAttentionNetwork
NeuralPlasticityMemory = ProductionNeuralPlasticityMemory
MultiActionHead = ProductionMultiActionHead
create_bio_module_spec = create_production_bio_module_spec

# Export public API
__all__ = [
    # Main module
    'BioInspiredRLModule',
    'ProductionUnifiedBioInspiredRLModule',

    # Components
    'PheromoneAttentionNetwork',
    'ProductionPheromoneAttentionNetwork',
    'NeuralPlasticityMemory',
    'ProductionNeuralPlasticityMemory',
    'MultiActionHead',
    'ProductionMultiActionHead',

    # Factory function
    'create_bio_module_spec',
    'create_production_bio_module_spec'
]
