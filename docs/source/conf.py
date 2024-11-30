# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DataLib'
copyright = '2024, Ahmed Hamila'
author = 'Ahmed Hamila'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Include links to highlighted source code
    'sphinx.ext.githubpages',  # Enable publishing on GitHub Pages (optional)
]

# Enable Napoleon for better Google/NumPy docstring support
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Recommended theme for better readability
html_static_path = ['_static']

# -- Add your project directory to sys.path ----------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))  # Adjust path to your source directory
