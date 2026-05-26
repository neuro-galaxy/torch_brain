from importlib import import_module
from pathlib import Path
import jinja2

import temporaldata

"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to the module's __api_ref__. Each module's
__api_ref__ consists of the following components:

description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - template (optional): the autosummary stub template to use for the section's
      entries, defaulting to ``base.rst``,
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
e.g., `temporaldata.interval`, the hook would be `interval_ref`; for a section, e.g.,
"Functions" under `temporaldata.concat`, the hook would be `concat_ref-functions`.
However, note that a better way is to refer using the :mod: directive, e.g.,
:mod:`temporaldata.interval` for the module. Only in case that a section is not a
particular submodule does the hook become useful.
"""


# Modules to include in API reference.
API_MODS = [
    "temporaldata",
]

API_REFERENCE = {m: import_module(m).__api_ref__ for m in API_MODS}


def build_api_rst():
    here = Path(__file__).parent
    generated = here / "generated"
    (generated / "api").mkdir(parents=True, exist_ok=True)

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
        with open(here / template["template_path"], "r") as f:
            t = jinja2.Template(f.read())

        # Render the template and write to the target
        with open(here / template["target_path"], "w") as f:
            f.write(t.render(**template["kwargs"]))
