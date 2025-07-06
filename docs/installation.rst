
# File: docs/installation.rst
"""
Installation instructions for different use cases.
"""

Installation Guide
==================

Nature MARL supports multiple installation methods depending on your use case.

🚀 Quick Installation
--------------------

For most users, install directly from PyPI:

.. code-block:: bash

   pip install nature-marl

This installs the core package with all required dependencies.

🔧 Development Installation
--------------------------

For development, testing, or contributing:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/nature-marl.git
   cd nature-marl

   # Install in development mode with all dependencies
   pip install -e ".[dev,docs,viz,gpu]"

   # Install pre-commit hooks
   pre-commit install

   # Run tests to verify installation
   pytest tests/ -v

📦 Installation Options
-----------------------

Nature MARL provides several optional dependency groups:

**Core Installation** (minimal dependencies):

.. code-block:: bash

   pip install nature-marl

**With Development Tools**:

.. code-block:: bash

   pip install "nature-marl[dev]"

**With Visualization Support**:

.. code-block:: bash

   pip install "nature-marl[viz]"

**With GPU Monitoring**:

.. code-block:: bash

   pip install "nature-marl[gpu]"

**Complete Installation** (all features):

.. code-block:: bash

   pip install "nature-marl[dev,docs,viz,gpu]"

🖥️ System Requirements
----------------------

**Minimum Requirements**:
- Python 3.8+
- 4GB RAM
- 2GB disk space

**Recommended Requirements**:
- Python 3.10+
- 8GB+ RAM
- GPU with 4GB+ VRAM (optional but recommended)
- 10GB+ disk space

**Supported Platforms**:
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS (10.15+)
- Windows (10+)

🐍 Python Environment Setup
---------------------------

We recommend using a virtual environment:

**Using conda**:

.. code-block:: bash

   conda create -n nature-marl python=3.10
   conda activate nature-marl
   pip install nature-marl

**Using venv**:

.. code-block:: bash

   python -m venv nature-marl-env
   source nature-marl-env/bin/activate  # On Windows: nature-marl-env\Scripts\activate
   pip install nature-marl

🚗 GPU Support
--------------

For GPU acceleration:

.. code-block:: bash

   # Install PyTorch with CUDA support first
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Then install Nature MARL
   pip install "nature-marl[gpu]"

Verify GPU installation:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")

✅ Verification
---------------

Verify your installation:

.. code-block:: python

   import nature_marl
   print(f"Nature MARL version: {nature_marl.__version__}")

   # Run basic functionality test
   from nature_marl.utils.debug_utils import run_installation_test
   run_installation_test()

Or use the command-line interface:

.. code-block:: bash

   nature-marl --version
   nature-marl test --quick

🔧 Troubleshooting
------------------

**Common Issues**:

**Ray Installation Issues**:

.. code-block:: bash

   # Reinstall Ray if you encounter issues
   pip uninstall ray
   pip install "ray[rllib]==2.8.0"

**Gymnasium Compatibility**:

.. code-block:: bash

   # Ensure compatible Gymnasium version
   pip install "gymnasium>=0.28.1,<0.30.0"

**GPU Not Detected**:

.. code-block:: bash

   # Check CUDA installation
   nvidia-smi

   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Import Errors**:

.. code-block:: bash

   # Clear Python cache and reinstall
   pip uninstall nature-marl
   pip cache purge
   pip install nature-marl

🆘 Getting Help
---------------

If you encounter issues:

1. **Check our FAQ**: `docs.nature-marl.org/faq`
2. **Search existing issues**: `github.com/your-org/nature-marl/issues`
3. **Create a new issue**: Include system info and error messages
4. **Join discussions**: `github.com/your-org/nature-marl/discussions`

**Include this information when reporting issues**:

.. code-block:: python

   import nature_marl
   nature_marl.utils.debug_utils.print_system_info()

This will output system configuration, dependency versions, and hardware information
to help diagnose problems.
