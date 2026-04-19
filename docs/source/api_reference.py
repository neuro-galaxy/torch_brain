from importlib import import_module
from pathlib import Path
import jinja2

import torch_brain
import torch_brain.dataset
import torch_brain.nn
import torch_brain.registry
import torch_brain.data.collate

"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to the modules's __api_ref__. Each module's
__api_ref__ consists of the following components:

description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_title_2 }}                                                       |
|     ---------------------                                                       |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `torch_brain.dataset`, the hook would be `dataset_ref`; for a
section, e.g., "Mixins" under `torch_brain.dataset`, the hook would be
`dataset_ref-mixins`. However, note that a better way is to refer using the :mod: directive,
e.g., :mod:`torch_brain.dataset` for the module. Only in case that a section
is not a particular submodule does the hook become useful.
"""


# Modules to include in API reference.
# Each entry: (module_dotted_name, section_title, members_to_exclude)
API_MODS = [
    "torch_brain.dataset",
    "torch_brain.data",
    "torch_brain.data.sampler",
    "torch_brain.transforms",
    "torch_brain.nn",
    "torch_brain.models",
    "torch_brain.registry",
    "torch_brain.utils",
]

API_REFERENCE = {m: import_module(m).__api_ref__ for m in API_MODS}


def build_api_rst():
    import importlib
    import pathlib

    generated = pathlib.Path(__file__).parent / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "api").mkdir(exist_ok=True)

    # rst_templates
    # kwargs: args to pass to jinja
    rst_templates: list[dict] = [
        {
            "template_path": "api/index.rst.template",
            "target_path": "generated/api/index.rst",
            "kwargs": {"API_REFERENCE": API_REFERENCE.items()},
        }
    ]

    for module in API_REFERENCE:
        rst_templates.append(
            {
                "template_path": "api/module.rst.template",
                "target_path": f"generated/api/{module}.rst",
                "kwargs": {"module": module, "module_info": API_REFERENCE[module]},
            }
        )

    for template in rst_templates:
        # Read the corresponding template file into jinja2
        with open(template["template_path"], "r") as f:
            t = jinja2.Template(f.read())

        # Render the template and write to the target
        with open(template["target_path"], "w") as f:
            f.write(t.render(**template["kwargs"]))
