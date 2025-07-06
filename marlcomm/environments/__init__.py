# marlcomm/environments/__init__.py
''' MARLCOMM Environments Module '''



try:
    from .emergence_environments import CommunicationEnv
except ImportError:
    print("Warning: Could not import CommunicationEnv")
    CommunicationEnv = None

__all__ = ["CommunicationEnv"]
