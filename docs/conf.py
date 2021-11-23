# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

for x in os.walk('../../heterogeneoushmm'):
    sys.path.insert(0, x[0])

# The main toctree document.
main_doc = "contents"

# -- Project information -----------------------------------------------------

project = 'HeterogeneousHMM'
copyright = '2021, Fernando Moreno Pino, Emese Sukei'
author = 'Fernando Moreno Pino, Emese Sukei'

# The full version, including alpha/beta/rc tags
version = release = '1.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

# -- Options for extensions --------------------------------------------------

autodoc_default_flags = ['members', 'inherited-members', 'show-inheritance']
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    'special-members': '__init__',
}

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

napoleon_use_ivar = True
napolean_use_param = True
napoleon_use_rtype = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**tests**', '**spi**']


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
htmlhelp_basename = 'pyhhmm_doc'
