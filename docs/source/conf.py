# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys

project = 'most-queue'
copyright = '2025, xabarov'
author = 'xabarov'
release = '1.51'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0,  os.path.abspath('../most_queue'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'autodocsumm',
              'sphinx.ext.coverage']

auto_doc_default_options = {'autosummary': True}
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
