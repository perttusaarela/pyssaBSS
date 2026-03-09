# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyssaBSS'
copyright = '2026, Perttu Saarela'
author = 'Perttu Saarela'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # generates docs from docstrings
    'sphinx.ext.napoleon',          # parses NumPy/Google style docstrings
    'sphinx.ext.viewcode',          # adds [source] links to each function
    'sphinx.ext.autosummary',       # generates summary tables
    'sphinx_autodoc_typehints',     # pulls type hints into docs
    'myst_parser',                  # lets you write .md files instead of .rst
]

# Napoleon settings — since you'll use NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Modules to exclude from documentation
exclude_from_autodoc = ['pyssaBSS.joint_diag']

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,  # set True to include functions with no docstring
    'show-inheritance': True,
}
autosummary_generate = True  # auto-generates stub .rst files

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
