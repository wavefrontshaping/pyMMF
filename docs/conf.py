# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'Your Project Name'
author = 'Your Name'
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_material'
html_theme_options = {
    'nav_title': 'Your Project Name',
    'color_primary': 'blue',
    'color_accent': 'light-blue',
    'repo_url': 'https://github.com/yourusername/yourproject/',
    'repo_name': 'yourproject',
    'globaltoc_depth': 2,
    'globaltoc_collapse': True,
    'globaltoc_includehidden': True,
}

html_static_path = ['_static']
