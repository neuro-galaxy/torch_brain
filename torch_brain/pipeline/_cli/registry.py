"""The registry of community brainsets plugins (``torch_brain/pipeline/plugins.toml``).

The registry is metadata only: it lets the CLI and the docs show plugins that are not
installed, and reserves their namespaces. The CLI never installs or runs a plugin
from the registry, it only prints the command to install it.
"""

import re
from dataclasses import dataclass, field
from importlib.resources import files

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python <3.11
    import tomli as tomllib  # type: ignore[import-not-found]

from .plugins import BUILTIN_NAMESPACE, NAMESPACE_PATTERN

REGISTRY_PATH = files("torch_brain.pipeline") / "plugins.toml"

_URL_PATTERN = re.compile(r"^https://\S+$")
_BRAINSET_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# key -> (type, required)
_PLUGIN_KEYS = {
    "package": (str, True),
    "description": (str, True),
    "install": (str, True),
    "repository": (str, True),
    "license": (str, True),
    "maintainers": (list, True),
    "brainsets": (list, True),
    "docs": (str, False),
    "modalities": (list, False),
}
_BRAINSET_KEYS = {
    "id": (str, True),
    "description": (str, True),
    "source": (str, False),
}


@dataclass
class RegisteredBrainset:
    id: str
    description: str
    source: str | None = None


@dataclass
class RegisteredPlugin:
    namespace: str
    package: str
    description: str
    install: str
    repository: str
    license: str
    maintainers: list[str]
    brainsets: list[RegisteredBrainset]
    docs: str | None = None
    modalities: list[str] = field(default_factory=list)


class RegistryError(ValueError):
    """Raised when ``plugins.toml`` does not follow the registry schema."""


def _check_keys(table: dict, schema: dict, where: str) -> list[str]:
    errors = []
    for key, (typ, required) in schema.items():
        if key not in table:
            if required:
                errors.append(f"{where}: missing required key '{key}'")
        elif not isinstance(table[key], typ):
            errors.append(f"{where}: '{key}' must be a {typ.__name__}")
    for key in table.keys() - schema.keys():
        errors.append(f"{where}: unknown key '{key}'")
    return errors


def validate_registry(data: dict) -> list[str]:
    """Return every schema violation in the parsed registry (empty if valid)."""
    errors = []
    for namespace, entry in data.items():
        where = f"[{namespace}]"
        if not NAMESPACE_PATTERN.match(namespace):
            errors.append(
                f"{where}: namespace must only contain lowercase letters, digits "
                "and underscores"
            )
        if namespace == BUILTIN_NAMESPACE:
            errors.append(f"{where}: the namespace '{BUILTIN_NAMESPACE}' is reserved")
        if not isinstance(entry, dict):
            errors.append(f"{where}: must be a table")
            continue

        errors += _check_keys(entry, _PLUGIN_KEYS, where)
        for key in ("repository", "docs"):
            value = entry.get(key)
            if isinstance(value, str) and not _URL_PATTERN.match(value):
                errors.append(f"{where}: '{key}' must be an https:// URL")
        for key in ("maintainers", "modalities"):
            value = entry.get(key)
            if isinstance(value, list) and not all(isinstance(v, str) for v in value):
                errors.append(f"{where}: '{key}' must be a list of strings")
        if entry.get("maintainers") == []:
            errors.append(f"{where}: 'maintainers' must not be empty")

        brainsets = entry.get("brainsets")
        if not isinstance(brainsets, list):
            continue
        if not brainsets:
            errors.append(f"{where}: 'brainsets' must not be empty")
        seen = set()
        for i, brainset in enumerate(brainsets):
            bwhere = f"[[{namespace}.brainsets]] #{i + 1}"
            if not isinstance(brainset, dict):
                errors.append(f"{bwhere}: must be a table")
                continue
            errors += _check_keys(brainset, _BRAINSET_KEYS, bwhere)
            brainset_id = brainset.get("id")
            if isinstance(brainset_id, str):
                if not _BRAINSET_ID_PATTERN.match(brainset_id):
                    errors.append(
                        f"{bwhere}: id '{brainset_id}' must be a valid identifier"
                    )
                if brainset_id in seen:
                    errors.append(f"{bwhere}: duplicate id '{brainset_id}'")
                seen.add(brainset_id)
            source = brainset.get("source")
            if isinstance(source, str) and not _URL_PATTERN.match(source):
                errors.append(f"{bwhere}: 'source' must be an https:// URL")
    return errors


def parse_registry(data: dict) -> dict[str, RegisteredPlugin]:
    """Build registry entries from parsed TOML.

    Raises:
        RegistryError: if the registry does not follow the schema.
    """
    errors = validate_registry(data)
    if errors:
        raise RegistryError("Invalid plugin registry:\n  " + "\n  ".join(errors))

    ret = {}
    for namespace, entry in data.items():
        entry = dict(entry)
        brainsets = [RegisteredBrainset(**b) for b in entry.pop("brainsets")]
        ret[namespace] = RegisteredPlugin(
            namespace=namespace, brainsets=brainsets, **entry
        )
    return ret


def load_registry_data(path=None) -> dict:
    path = REGISTRY_PATH if path is None else path
    with path.open("rb") as f:
        return tomllib.load(f)


def load_registry(path=None) -> dict[str, RegisteredPlugin]:
    """Load the plugin registry bundled with torch_brain, keyed by namespace."""
    return parse_registry(load_registry_data(path))


def install_hint(plugin: RegisteredPlugin) -> str:
    """The message telling a user how to install a registered plugin."""
    return (
        f"Install it with:\n"
        f'  uv add "{plugin.install}"\n'
        f"or:\n"
        f'  pip install "{plugin.install}"\n'
        f"'{plugin.namespace}' is a community plugin, maintained by its authors "
        f"({plugin.repository}), not by torch_brain."
    )
