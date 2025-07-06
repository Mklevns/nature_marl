# File: nature_marl/training/examples/custom_learner_example.py
#!/usr/bin/env python3

"""
Custom PPO Learner that integrates nature-inspired communication loss
with the standard PPO objective.
"""

import torch
from typing import Dict, Any
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import MultiAgentBatch


class NatureInspiredPPOLearner(PPOTorchLearner):
    """
    Enhanced PPO Learner that incorporates nature-inspired communication losses
    alongside the standard PPO objective.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Communication loss coefficients
        self.comm_loss_coeff = self.config.get("comm_loss_coeff", 0.1)
        self.comm_entropy_coeff = self.config.get("comm_entropy_coeff", 0.01)
        self.comm_sparsity_coeff = self.config.get("comm_sparsity_coeff", 0.001)
        self.comm_diversity_coeff = self.config.get("comm_diversity_coeff", 0.01)

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: str,
        config: LearnerHyperparameters,
        batch: MultiAgentBatch,
        fwd_out: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute combined PPO + communication loss for the module.
        """
        # Get standard PPO loss
        ppo_loss_dict = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out
        )

        # Get the module
        module = self.module[module_id]

        # Add communication loss if available
        if hasattr(module, 'get_communication_loss'):
            comm_loss = module.get_communication_loss(
                entropy_coeff=self.comm_entropy_coeff,
                sparsity_coeff=self.comm_sparsity_coeff,
                diversity_coeff=self.comm_diversity_coeff
            )

            # Add communication loss to total loss
            total_loss = ppo_loss_dict["total_loss"] + self.comm_loss_coeff * comm_loss

            # Update loss dictionary
            ppo_loss_dict.update({
                "total_loss": total_loss,
                "communication_loss": comm_loss,
                "communication_entropy": module.comm_entropy if hasattr(module, 'comm_entropy') else 0.0,
                "communication_usage": module.comm_usage if hasattr(module, 'comm_usage') else 0.0,
                "attention_diversity": module.attention_diversity if hasattr(module, 'attention_diversity') else 0.0
            })

        return ppo_loss_dict


# Configuration helper for the custom learner
def get_nature_inspired_ppo_config(base_config: dict) -> dict:
    """
    Enhance a PPO config to use the nature-inspired learner.

    Args:
        base_config: Base PPO configuration dictionary

    Returns:
        Enhanced configuration with custom learner
    """
    enhanced_config = base_config.copy()

    # Add communication loss coefficients
    enhanced_config.update({
        "comm_loss_coeff": 0.1,
        "comm_entropy_coeff": 0.01,
        "comm_sparsity_coeff": 0.001,
        "comm_diversity_coeff": 0.01,
    })

    return enhanced_config


# Usage in configuration
def create_enhanced_ppo_config(num_agents: int = 8):
    """Create a complete PPO config with nature-inspired communication."""
    from ray.rllib.algorithms.ppo import PPOConfig

    config = (
        PPOConfig()
        .learners(
            learner_class=NatureInspiredPPOLearner,
            num_learners=1,
            num_gpus_per_learner=1
        )
        .training(
            # Standard PPO params
            train_batch_size=8192,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,

            # Communication loss coefficients
            comm_loss_coeff=0.1,
            comm_entropy_coeff=0.01,
            comm_sparsity_coeff=0.001,
            comm_diversity_coeff=0.01,
        )
    )

    return config
