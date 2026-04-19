import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))

import torch_brain

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
]

autosummary_generate = []

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]

add_module_names = False
autodoc_member_order = "bysource"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
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
html_logo = "_static/torch_brain_logo.png"
html_favicon = "_static/torch_brain_logo.png"

# Modules to include in API reference.
# Each entry: (module_dotted_name, section_title, members_to_exclude)
_API_MODULES = [
    ("torch_brain.dataset", "torch_brain.dataset", []),
    ("torch_brain.data.collate", "torch_brain.data.collate", []),
    ("torch_brain.data.sampler", "torch_brain.data.sampler", []),
    ("torch_brain.data.dataset", "torch_brain.data.dataset", []),
    ("torch_brain.models", "torch_brain.models", []),
]


def _build_api_rst(app):
    import importlib
    import pathlib

    src = pathlib.Path(app.srcdir)
    generated = src / "generated"
    generated.mkdir(exist_ok=True)

    index = [
        "API Reference",
        "=============",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for mod_name, _, _ in _API_MODULES:
        index.append(f"   {mod_name}")
    generated.joinpath("api.rst").write_text("\n".join(index) + "\n")
    autosummary_generate.append("generated/api")

    for mod_name, title, exclude in _API_MODULES:
        autosummary_generate.append(f"generated/{mod_name}")

        mod = importlib.import_module(mod_name)
        members = [m for m in (getattr(mod, "__all__", []) or []) if m not in exclude]
        page = [
            title,
            "=" * len(title),
            "",
            f".. automodule:: {mod_name}",
            "   :no-members:",
            "",
            "Module Reference",
            "----------------",
            "",
            ".. autosummary::",
            "   :toctree: .",
            "   :nosignatures:",
            "",
        ]
        for m in members:
            page.append(f"   ~{mod_name}.{m}")
        generated.joinpath(f"{mod_name}.rst").write_text("\n".join(page) + "\n")


def setup(app):
    app.connect("builder-inited", _build_api_rst, priority=100)
