# Configuration file for the Sphinx documentation builder.


import sys
import os


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))
import recommonmark
from recommonmark.transform import AutoStructify
from sphinx_pyproject import SphinxConfig
import sphinx_material

sys.path.insert(0, os.path.abspath("./sphinx_ext"))

# -- Project information -----------------------------------------------------

config = SphinxConfig("../pyproject.toml", globalns=globals())

project_name = "pyMMF"
copyright = "2024, " + author


# project = "pySLM2"
# author
version
# description
# copyright = '2023, Chung-You (Gilbert) Shih'
# release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    # "recommonmark",
    # "m2r2",
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "custom",
    "numpydoc",
    # "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "venv",
    "**.ipynb_checkpoints",
    "test*.ipynb",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
}
autodoc_typehints = "description"

sphinx_togglebutton_selector = ".nboutput, .nbinput"

# Napoleon settings
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True

mathjax_path = "_static/scipy-mathjax/MathJax.js?config=scipy-mathjax"

# -- Options for HTML output -------------------------------------------------

# -- Options for source suffix ------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

# html_theme_path = sphinx_material.html_theme_path()
# html_context = sphinx_material.get_html_context()

# html_theme = "sphinx_material"
# html_theme_options = {
#     "nav_title": f"{project_name} v{version}",
#     "color_primary": "blue",
#     "color_accent": "light-blue",
#     "base_url": "http://127.0.0.1:5500/docs/_build/html/",
#     "repo_url": "https://github.com/wavefrontshaping/pyMMF/",
#     "repo_name": project_name,
#     "repo_type": "github",
#     "globaltoc_depth": 2,
#     "globaltoc_collapse": True,
#     "globaltoc_includehidden": True,
#     "html_minify": True,
#     "html_prettify": True,
#     # "table_classes": ["plain"],
# }
# html_sidebars = {
#     "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
# }

html_theme = "pydata_sphinx_theme"


html_title = f"{project_name} v{version}"

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/wavefrontshaping/pymmf",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "X",
            # URL where the link will redirect
            "url": "https://twitter.com/WFShaping",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-twitter",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "wavefrontshaping.net",
            # URL where the link will redirect
            "url": "https://wavefrontshaping.net",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fas fa-link",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
    ],
}

# remove left section sidebar
# see https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html
html_sidebars = {"*": []}
# html_sidebars = {
#     "community/index": [
#         "sidebar-nav-bs",
#         "custom-template",
#     ],
#     "whats-new": [],
# }

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

templates_path = ["_templates"]


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            #'url_resolver': lambda url: github_doc_root + url,
            "auto_toc_tree_section": "Contents",
            "enable_math": False,
            "enable_inline_math": False,
            "enable_eval_rst": True,
        },
        True,
    )
    app.add_transform(AutoStructify)
