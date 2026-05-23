from pathlib import Path
import json
import nbformat


def main():
    src = Path(__file__).parent / "nlb_maze_minimal_example.ipynb"
    dest = (
        Path(__file__).parent.parent.parent
        / "generated/notebooks/nlb_maze_minimal_example.ipynb"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    colab_path = (
        "https://colab.research.google.com/github/neuro-galaxy/torch_brain/blob/main/"
        "docs/source/notebooks/nlb_maze_minimal_example/nlb_maze_minimal_example.ipynb"
    )

    with src.open("r") as f:
        notebook = json.load(f)

    # Hide some cells
    hide_cell_patterns = (
        "class Linear(nn.Module)",
        "class GRU(nn.Module)",
        "class TCN(nn.Module)",
    )
    hide_output_patterns = (
        "!pip install",
        "!brainsets prepare",
    )
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        tag = None
        if any(p in src for p in hide_cell_patterns):
            tag = "hide-cell"
        elif any(p in src for p in hide_output_patterns):
            tag = "hide-output"
        if tag:
            tags = cell.setdefault("metadata", {}).setdefault("tags", [])
            if tag not in tags:
                tags.append(tag)

    # Add links to google collab at the start and end
    colab_link_md = (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_path})\n"
    )
    notebook["cells"] = (
        [nbformat.v4.new_markdown_cell(colab_link_md)]
        + notebook.get("cells", [])
        + [nbformat.v4.new_markdown_cell(colab_link_md)]
    )
    dest.write_text(json.dumps(notebook))
