# File: docs/conf.py
"""
Sphinx configuration for Bio-Inspired MARL documentation.

This configuration generates comprehensive documentation including:
- API reference with auto-generated docstrings
- Bio-inspired concept explanations
- Usage tutorials and examples
- Performance benchmarks
- Research applications
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = 'Nature-Inspired Multi-Agent Reinforcement Learning'
copyright = '2024, Nature MARL Team'
author = 'Nature MARL Team'
release = '1.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# Auto-documentation settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
}

# Source file configuration
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980b9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# LaTeX output configuration
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'maketitle': '',
}

latex_documents = [
    (master_doc, 'NatureMARL.tex', 'Nature-Inspired Multi-Agent RL Documentation',
     'Nature MARL Team', 'manual'),
]

# Manual page output
man_pages = [
    (master_doc, 'nature-marl', 'Nature-Inspired Multi-Agent RL Documentation',
     [author], 1)
]
