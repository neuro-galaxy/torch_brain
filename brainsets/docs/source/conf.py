import os
import sys
import datetime

import brainsets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))

author = "neuro-galaxy Team"
project = "brainsets"
version = brainsets.__version__
copyright = f"{datetime.datetime.now().year}, {author}"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_inline_tabs",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "custom_autosummary",  # local extension to expand `{% for name in module._list %}` blocks in rst files to render api tables and generate stub pages.
]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]

add_module_names = True
autodoc_member_order = "bysource"
autosummary_generate = True

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "temporaldata": ("https://temporaldata.readthedocs.io/en/latest/", None),
    "torch_brain": ("https://torch-brain.readthedocs.io/en/latest/", None),
}

myst_enable_extensions = [
    "html_admonition",
    "html_image",
]

pygments_style = "default"

html_css_files = [
    "style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
]

html_js_files = [
    "citation.js",
]
html_copy_source = False
html_show_sourcelink = True
html_logo = "_static/brainsets_logo.png"
html_favicon = "_static/brainsets_logo.png"
