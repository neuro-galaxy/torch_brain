"""Builds the "Community brainsets" page from torch_brain/pipeline/plugins.toml."""

import pathlib

import jinja2

from torch_brain.pipeline._cli.registry import load_registry


def build_community_brainsets_rst():
    here = pathlib.Path(__file__).parent
    generated = here / "generated"
    generated.mkdir(exist_ok=True)

    plugins = [load_registry()[ns] for ns in sorted(load_registry())]
    template = jinja2.Template(
        (here / "community" / "community_brainsets.rst.template").read_text()
    )
    (generated / "community_brainsets.rst").write_text(template.render(plugins=plugins))
