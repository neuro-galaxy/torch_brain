"""
Monkey-patches ``sphinx.ext.autosummary`` to handle
``{% for name in module.attr %}`` lines inside ``.. autosummary::`` blocks.

Patches both stub generation (find_autosummary_in_lines) and HTML table

Adapted from https://github.com/pyg-team/pyg_sphinx_theme/blob/master/pyg_sphinx_theme/extension/pyg.py
"""

import importlib
import re
from typing import List, Optional

from docutils.statemachine import StringList

import sphinx.ext.autosummary as autosummary_ext
import sphinx.ext.autosummary.generate as autosummary


_list_arg_re = re.compile(r"^\s*{%\s*for\s+\S+\s+in\s+(.*?)\s*%}\s*$")


def _normalize_entry(entry) -> str:
    """Normalize an autosummary list entry to a plain object name."""
    name = str(entry).strip()
    if name.startswith("~"):
        name = name[1:]
    return name


def _expand_for_lines(lines):
    """Expand ``{% for name in module.attr %}`` lines to actual names."""
    expanded = []
    for line in lines:
        m = _list_arg_re.match(line)
        if m:
            obj_path = m.group(1).strip()
            module_name, obj_name = obj_path.rsplit(".", maxsplit=1)
            mod = importlib.import_module(module_name)
            indent = len(line) - len(line.lstrip())
            for entry in getattr(mod, obj_name):
                expanded.append(" " * indent + _normalize_entry(entry))
        else:
            expanded.append(line)
    return expanded


_original_autosummary_run = autosummary_ext.Autosummary.run


# Custom: expand {% for <var> in <module.attr> %} into toctree table
def _patched_autosummary_run(self):
    """Expand {% for %} lines in directive content before building the table."""
    expanded = _expand_for_lines(list(self.content))
    self.content = StringList(expanded, source="<custom_autosummary expanded>")
    return _original_autosummary_run(self)


def monkey_patch_find_autosummary_in_lines(
    lines: List[str],
    module: str = None,
    filename: str = None,
) -> autosummary.AutosummaryEntry:
    import os.path as osp

    autosummary_re = re.compile(r"^(\s*)\.\.\s+autosummary::\s*")
    automodule_re = re.compile(r"^\s*\.\.\s+automodule::\s*([A-Za-z0-9_.]+)\s*$")
    module_re = re.compile(r"^\s*\.\.\s+(current)?module::\s*([a-zA-Z0-9_.]+)\s*$")
    autosummary_item_re = re.compile(r"^\s+(~?[_a-zA-Z][a-zA-Z0-9_.]*)\s*.*?")
    recursive_arg_re = re.compile(r"^\s+:recursive:\s*$")
    toctree_arg_re = re.compile(r"^\s+:toctree:\s*(.*?)\s*$")
    template_arg_re = re.compile(r"^\s+:template:\s*(.*?)\s*$")
    list_arg_re = re.compile(r"^\s+{% for\s*(.*?)\s*in\s*(.*?)\s*%}$")

    documented: list[autosummary.AutosummaryEntry] = []

    recursive = False
    toctree: Optional[str] = None
    template = None
    curr_module = module
    in_autosummary = False
    base_indent = ""

    for line in lines:
        if in_autosummary:
            m = recursive_arg_re.match(line)
            if m:
                recursive = True
                continue

            m = toctree_arg_re.match(line)
            if m:
                toctree = m.group(1)
                if filename:
                    toctree = osp.join(osp.dirname(filename), toctree)
                continue

            m = template_arg_re.match(line)
            if m:
                template = m.group(1).strip()
                continue
            # Custom: expand {% for <var> in <module.attr> %} into autosummary entries
            m = list_arg_re.match(line)
            if m:
                obj_name = m.group(2).strip()
                module_name, obj_name = obj_name.rsplit(".", maxsplit=1)
                mod = importlib.import_module(module_name)
                for entry in getattr(mod, obj_name):
                    documented.append(
                        autosummary.AutosummaryEntry(
                            f"{module_name}.{_normalize_entry(entry)}",
                            toctree,
                            template,
                            recursive,
                        )
                    )
                continue
            if line.strip().startswith(":"):
                continue

            m = autosummary_item_re.match(line)
            if m:
                name = m.group(1).strip()
                if name.startswith("~"):
                    name = name[1:]
                if curr_module and not name.startswith(f"{curr_module}."):
                    name = f"{curr_module}.{name}"
                documented.append(
                    autosummary.AutosummaryEntry(
                        name,
                        toctree,
                        template,
                        recursive,
                    )
                )
                continue

            if not line.strip() or line.startswith(f"{base_indent} "):
                continue

            in_autosummary = False

        m = autosummary_re.match(line)
        if m:
            in_autosummary = True
            base_indent = m.group(1)
            recursive = False
            toctree = None
            template = None
            continue

        m = automodule_re.search(line)
        if m:
            curr_module = m.group(1).strip()
            documented.extend(
                autosummary.find_autosummary_in_docstring(
                    curr_module,
                    filename=filename,
                )
            )
            continue

        m = module_re.match(line)
        if m:
            curr_module = m.group(2)
            continue

    return documented


def setup(app):
    autosummary.find_autosummary_in_lines = monkey_patch_find_autosummary_in_lines
    autosummary_ext.Autosummary.run = _patched_autosummary_run
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
