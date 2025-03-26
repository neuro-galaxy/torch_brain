import datetime

import torch_brain

author = "neuro-galaxy Team"
project = "torch_brain"
version = "0.0.1"  # torch_brain.__version__
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
    "bokeh.sphinxext.bokeh_plot",
    "sphinx_copybutton",
]

html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]

add_module_names = False
autodoc_member_order = "bysource"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    # "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "temporaldata": ("https://temporaldata.readthedocs.io/en/latest/", None),
}

myst_enable_extensions = [
    "html_admonition",
    "html_image",
]

pygments_style = "default"

bokeh_plot_pyfile_include_dirs = [
    "concepts/examples",
]
html_copy_source = False
html_show_sourcelink = True

html_copy_source = False
html_show_sourcelink = True
html_logo = "_static/torch_brain_logo.png"
html_favicon = "_static/torch_brain_logo.png"
