import sys
from pathlib import Path


def fetch_notebook():
    """Download the NLB Maze training notebook from Google Drive and inject a
    preamble cell with a link back to Colab. Skipped if already present —
    use ``make clean html`` to force a redownload. On network/parse failure
    a warning is emitted and the download is skipped so the docs build can
    still complete offline."""
    import json
    import socket
    import urllib.error
    import urllib.request

    import nbformat

    colab_id = "1r1vbxmqccgHz-6det9Bxk3Ld6xU8RZdk"
    dest = Path("generated/nlb_maze_train.ipynb")
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={colab_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
        notebook = json.loads(raw)
    except (urllib.error.URLError, socket.timeout, json.JSONDecodeError) as e:
        print(
            f"WARNING: could not fetch nlb_maze_train.ipynb from Google Drive "
            f"({type(e).__name__}: {e}). The nlb_maze_train page will be missing "
            f"from this build.",
            file=sys.stderr,
        )
        return

    colab_link_md = (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/drive/{colab_id})\n"
    )
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
    notebook["cells"] = (
        [nbformat.v4.new_markdown_cell(colab_link_md)]
        + notebook.get("cells", [])
        + [nbformat.v4.new_markdown_cell(colab_link_md)]
    )
    dest.write_text(json.dumps(notebook))
