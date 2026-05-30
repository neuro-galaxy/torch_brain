# from https://github.com/facebookresearch/hydra/blob/main/examples/plugins/example_searchpath_plugin/hydra_plugins/example_searchpath_plugin/example_searchpath_plugin.py
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra import initialize
from hydra.plugins.search_path_plugin import SearchPathPlugin

from hydra_plugins.searchpath_plugin import (
    KirbySearchPathPlugin,
)


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert KirbySearchPathPlugin.__name__ in [
        x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)
    ]


def test_config_installed() -> None:
    with initialize(version_base=None):
        config_loader = GlobalHydra.instance().config_loader()
        assert "perich_miller_population_2018" in config_loader.get_group_options(
            "dataset"
        )
        assert "poyo_1" in config_loader.get_group_options("model")
