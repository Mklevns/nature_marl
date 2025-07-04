#!/usr/bin/env python3
"""Ultra-minimal GPU test - just PyTorch and Ray."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Simple GPU computation
    a = torch.randn(1000, 1000).cuda()
    b = a @ a.T
    print(f"GPU computation successful!")
    print(f"Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Test Ray GPU allocation
import ray
ray.init(num_gpus=1, logging_level="ERROR")
print(f"\nRay GPU available: {ray.available_resources().get('GPU', 0)}")
ray.shutdown()

print("\n✅ Basic GPU functions work!")
print("\n🔧 To fix RLlib dependencies:")
print("   pip install opencv-python")
print("   pip install 'numpy==1.26.4'")
