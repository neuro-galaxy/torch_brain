from pathlib import Path


def fetch_notebook():
    """Download the NLB Maze training notebook from Google Drive and inject a
    preamble cell with a link back to Colab. Skipped if already present —
    use ``make clean html`` to force a redownload."""
    import json
    import urllib.request

    colab_id = "1r1vbxmqccgHz-6det9Bxk3Ld6xU8RZdk"
    dest = Path("generated/nlb_maze_train.ipynb")
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={colab_id}"
    with urllib.request.urlopen(url) as resp:
        notebook = json.loads(resp.read())

    collab_link = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/drive/{colab_id})\n",
        ],
    }
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

    notebook["cells"] = [collab_link] + notebook.get("cells", []) + [collab_link]
    dest.write_text(json.dumps(notebook))
