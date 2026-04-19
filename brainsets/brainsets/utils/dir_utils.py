import os
import pathlib
import shutil
from collections import defaultdict
import logging

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich.tree import Tree


def find_files_by_extension(folder_path, extension):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(root, file)


def make_directory(path, prompt_if_exists=False):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info("New directory [u]{}[/u] created.".format(path))
    else:
        logging.info("Directory [u]{}[/u] already exists.".format(path))
        if not prompt_if_exists:
            return

        rm_dir = Confirm.ask(
            "Do you want to delete the folder? "
            "[bold red blink]This will permanently remove its contents.[/bold red blink]",
            default=False,
        )

        if rm_dir:
            shutil.rmtree(path)
            os.makedirs(path)
            logging.info("New directory [u]{}[/u] created.".format(path))
        else:
            # kill process
            logging.info("Killing process.")
            exit()


def get_dir_tree(path):
    tree = Tree(
        f":open_file_folder: [link file://{path}]{path}",
        guide_style="bold bright_blue",
    )
    walk_directory(pathlib.Path(path), tree)
    return tree


def walk_directory(directory: pathlib.Path, tree: Tree) -> None:
    paths = sorted(
        pathlib.Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )

    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue

        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            try:
                create_time = path.stat().st_birthtime
                text_filename.append(f" ({create_time})", "blue")
            except:
                pass
            icon = defaultdict(
                lambda: "📄 ",
                py="🐍 ",
                cfg="🛠 ",
                pt="🔥 ",
                nwb="🧠 ",
                npy="🐍 ",
                mat="📊 ",
            )[path.suffix[1:]]
            tree.add(Text(icon) + text_filename)
