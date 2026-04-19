import datetime
import os
import sys
from pathlib import Path
from sphinx.util.typing import restify

import torch_brain

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("sphinxext"))


author = "neuro-galaxy Team"
project = "torch_brain"
version = torch_brain.__version__
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
    "bokeh.sphinxext.bokeh_plot",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.sass",
    # see sphinxext/
    "autoshortsummary",
]

autosummary_generate = True

# Compile scss files into css files using sphinxcontrib-sass
sass_src_dir, sass_out_dir = "scss", "generated/css/styles"
sass_targets = {
    f"{file.stem}.scss": f"{file.stem}.css"
    for file in Path(sass_src_dir).glob("*.scss")
}
Path("generated/css/").mkdir(exist_ok=True, parents=True)

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "torch_brain docs",
        "image_light": "_static/torch_brain_logo.png",
        "image_dark": "_static/torch_brain_logo.png",
    }
}
html_static_path = ["_static", "generated/css", "js"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]

add_module_names = True
autodoc_member_order = "bysource"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "h5py": ("http://docs.h5py.org/en/latest/", None),
    "temporaldata": ("https://temporaldata.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
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
html_favicon = "_static/torch_brain_logo.png"


from api_reference import build_api_rst

build_api_rst()


def add_js_css_files(app, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages.

    Note that `html_js_files` and `html_css_files` are included in all pages and
    should be used for the ones that are used by multiple pages. All page-specific
    JS and CSS files should be added here instead.
    """
    if pagename == "generated/api/index":
        # External: jQuery and DataTables
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file(
            "https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css"
        )
        # Internal: API search initialization and styling
        app.add_js_file("scripts/api-search.js")
        app.add_css_file("styles/api-search.css")


def _process_bases(app, name, obj, options, bases):
    # This shows torch.utils.data.dataset.Dataset as the base
    # without this, it would show up as "Dataset"
    bases[:] = [restify(b, "fully-qualified-except-typing") for b in obj.__bases__]


def setup(app):
    app.connect("autodoc-process-bases", _process_bases)
    # triggered just before the HTML for an individual page is created
    app.connect("html-page-context", add_js_css_files)
