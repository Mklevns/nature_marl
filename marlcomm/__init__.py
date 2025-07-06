 #marlcomm/utils/__init__.py":

"""Utility modules for MARLCOMM."""

# Import what exists
try:
    from .callbacks import BioInspiredCallbacks
except ImportError:
    print("Warning: Could not import callbacks")
    BioInspiredCallbacks = None

try:
    from .logging_config import setup_logging
except ImportError:
    print("Warning: Could not import logging_config")
    setup_logging = None

__all__ = ["BioInspiredCallbacks", "setup_logging"]
