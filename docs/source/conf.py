import os
import sys

# Insert the parent directory (the folder above `docs/`) so that Python can find vsf/
sys.path.insert(0, os.path.abspath('..'))

autodoc_mock_imports = [
    "OpenGL",
    "OpenGL.GL",
    "OpenGL.GLU",
    # any other modules that load OpenGL
]

# -- Project information
project = 'openvsf'
copyright = '2025, Shaoxiong Yao'
author = 'Shaoxiong Yao'
release = '0.0.1'

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output
html_theme = 'alabaster'
html_static_path = ['_static']