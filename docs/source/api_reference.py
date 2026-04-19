import torch_brain

# Modules to include in API reference.
# Each entry: (module_dotted_name, section_title, members_to_exclude)
_API_MODULES = [
    ("torch_brain.dataset", "torch_brain.dataset", []),
    ("torch_brain.data.collate", "torch_brain.data.collate", []),
    ("torch_brain.data.sampler", "torch_brain.data.sampler", []),
    ("torch_brain.data.dataset", "torch_brain.data.dataset", []),
    ("torch_brain.transforms", "torch_brain.transforms", []),
    ("torch_brain.models", "torch_brain.models", []),
    ("torch_brain.nn", "torch_brain.nn", []),
    ("torch_brain.registry", "torch_brain.registry", []),
]


def build_api_rst():
    import importlib
    import pathlib

    generated = pathlib.Path(__file__).parent / "generated"
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

    for mod_name, title, exclude in _API_MODULES:
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
            page.append(f"   {m}")
        generated.joinpath(f"{mod_name}.rst").write_text("\n".join(page) + "\n")
