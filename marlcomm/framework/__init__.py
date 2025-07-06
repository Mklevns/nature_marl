# marlcomm/framework/__init__.py:
"""MARLCOMM Framework Module."""

try:
    from .research import ResearchHypothesis
except ImportError:
    ResearchHypothesis = None

__all__ = ["ResearchHypothesis"]
