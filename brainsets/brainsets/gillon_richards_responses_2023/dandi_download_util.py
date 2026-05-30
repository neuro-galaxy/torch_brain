#!/usr/bin/env python

"""
dandi_download_util.py

This module contains functions for downloading the dataset from the Dandi 
archive.

Dandiset 000037 is the Credit Assignment Project dandiset. It comprises data 
for 50 sessions. The asset (session file) sizes are in the following ranges:
- Basic data (with everything required for most analyses): 
    130 MB to 1.7 GB per asset
    ~25 GB total
- Basic data + stimulus template images: 
    1.5 to 3.1 GB per asset
    ~100 GB total
- Basic data + stimulus template images + full imaging stack: 
    ~45 GB per asset
    ~2.3 TB total
    ~1.5 TB total (only sess 1-3 that passed QC, i.e., 33 total)

URL: https://gui.dandiarchive.org/#/dandiset/000037

Authors: Colleen Gillon

Date: Nov, 2022

Note: this code uses python 3.7.

"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from dandi import dandiapi
from dandi import download as dandi_download


# if running from the main directory
DEFAULT_MOUSE_DF_PATH = Path("mouse_df.csv")

# Published dandiset version
PUBLISHED_VERSION = "0.230426.0054"  # number derived from DOI


#############################################
def reformat_n(n):
    """
    reformat_n(n)

    Returns reformatted n argument, converting ranges to lists.

    Required args:
        - n (str):
            number or range (e.g., "1-1", "all")

    Returns:
        - n (list):
            number or range (e.g., [1, 2, 3], "all")
    """

    if isinstance(n, list):
        n = [int(i) for i in n]
    elif "-" in str(n):
        vals = str(n).split("-")
        if len(vals) != 2:
            raise ValueError("If n is a range, must have format 1-3.")
        st = int(vals[0])
        end = int(vals[1]) + 1
        n = list(range(st, end))
    elif n not in ["all", "any"]:
        if not str(n).isdigit():
            raise ValueError(f"'n' expected to be a digit, but found {n}.")
        n = [int(n)]

    return n


#############################################
def get_dandi_session_ids(
    mouse_df, mouse_n="all", sess_n="all", pass_fail="P", sort=True
):
    """
    get_dandi_session_ids(mouse_df)

    Returns list of Dandi session IDs that fit the specified criteria.

    Required args:
        - mouse_df (Path or pd.DataFrame): path to dataframe containing
                                           information on each session or
                                           dataframe itself

    Optional args:
        - mouse_n (int, str or list)  : mouse number(s) to retain,
                                        default: "all"
        - sess_n (int, str or list)   : session number(s) to retain
                                        default: "all"
        - pass_fail (str or list)     : pass/fail values to retain
                                        ("P", "F", "all")
                                        default: "P"

    Returns:
        - dandi_session_ids (list): Dandi session IDs that fit criteria
    """

    if isinstance(mouse_df, (str, Path)):
        if not Path(mouse_df).is_file():
            raise OSError(f"{mouse_df} does not exist.")
        mouse_df = pd.read_csv(mouse_df)

    criteria = ["runtype: production"]
    lines = mouse_df.loc[
        (mouse_df["runtype"] == "prod")
        & (mouse_df["sessid"] != 838633305)  # excluded session
    ]

    # retain lines that fit the session and mouse criteria
    col_names = ["mouse_n", "sess_n", "pass_fail"]
    col_vals = [mouse_n, sess_n, pass_fail]

    for name, vals in zip(col_names, col_vals):
        criteria.append(f"{name}: {vals}")
        if vals not in ["all", "any"]:
            if name in ["mouse_n", "sess_n"]:
                vals = reformat_n(vals)
            elif not isinstance(vals, list):
                vals = [vals]
            lines = lines.loc[(lines[name].isin(vals))]

    if len(lines) == 0:
        raise ValueError(
            f"No sessions fit the combined criteria: {', '.join(criteria)}"
        )

    dandi_session_ids = lines["sessid"].tolist()
    if sort:
        dandi_session_ids = sorted(dandi_session_ids)

    return dandi_session_ids


#############################################
def get_dandiset_asset_urls(
    dandiset_id="000037",
    version=PUBLISHED_VERSION,
    asset_sessids="all",
    incl_stim_templates=False,
    incl_full_stacks=False,
):
    """
    get_dandiset_asset_urls
    """

    client = dandiapi.DandiAPIClient()
    dandi = client.get_dandiset(dandiset_id, version)

    if asset_sessids != "all":
        if isinstance(asset_sessids, list):
            asset_sessids = [str(asset_sessid) for asset_sessid in asset_sessids]
        else:
            asset_sessids = [str(asset_sessids)]

    if incl_full_stacks and not incl_stim_templates:
        warnings.warn(
            "The files that include the full stacks also include the "
            "stimulus templates."
        )
        incl_stim_templates = True

    asset_urls = []
    selected_asset_sessids = []
    for asset in dandi.get_assets():
        asset_path = Path(asset.path)
        asset_sessid = asset_path.parts[1].split("_")[1].replace("ses-", "")
        if asset_sessids != "all" and asset_sessid not in asset_sessids:
            continue

        stim_templates_included = "+image" in asset_path.parts[1]
        if stim_templates_included != incl_stim_templates:
            continue

        full_stacks_included = "obj-raw" in asset_path.parts[1]
        if full_stacks_included != incl_full_stacks:
            continue

        asset_urls.append(asset.download_url)
        selected_asset_sessids.append(asset_sessid)

    asset_urls = [asset_urls[i] for i in np.argsort(selected_asset_sessids)]

    if len(asset_urls) == 0:
        raise RuntimeError("No dandiset assets found that meet the criteria.")

    return asset_urls


#############################################
def download_dandiset_assets(
    dandiset_id="000037",
    version=PUBLISHED_VERSION,
    output=".",
    incl_stim_templates=False,
    incl_full_stacks=False,
    sess_ns="all",
    mouse_ns="all",
    excluded_sess=True,
    mouse_df=DEFAULT_MOUSE_DF_PATH,
    dry_run=False,
    n_jobs=6,
):

    dandiset_id = f"{int(dandiset_id):06}"  # ensure correct ID formatting

    asset_sessids = "all"
    if not (excluded_sess and sess_ns in ["all", "any"] and mouse_ns in ["all", "any"]):
        if dandiset_id != "000037":
            raise NotImplementedError(
                "Selecting assets based on session and mouse numbers is only "
                "implemented for dandiset 000037."
            )

        pass_fail = "all" if excluded_sess else "P"
        asset_sessids = get_dandi_session_ids(
            mouse_df, mouse_n=mouse_ns, sess_n=sess_ns, pass_fail=pass_fail, sort=True
        )

    print("Identifying the URLs of dandi assets to download...")
    dandiset_urls = get_dandiset_asset_urls(
        dandiset_id,
        version=version,
        asset_sessids=asset_sessids,
        incl_stim_templates=incl_stim_templates,
        incl_full_stacks=incl_full_stacks,
    )

    action_str = "Identified" if dry_run else "Downloading"
    end_str = ". Run without '--dry_run' to download." if dry_run else "..."

    print(
        f"{action_str} {len(dandiset_urls)} assets from "
        f"dandiset {dandiset_id}{end_str}"
    )

    if dry_run:
        return

    try:
        dandi_download.download(dandiset_urls, output, jobs=n_jobs, existing="refresh")
    except NotImplementedError as err:
        if "multiple URLs not supported" not in str(err):
            raise err
        if n_jobs != 1:
            warnings.warn(
                "Downloading data sequentially. Upgrade Dandi to version "
                "0.50 or above to download from multiple URLs in parallel."
            )
        for dandiset_url in dandiset_urls:
            dandi_download.download(
                dandiset_url, output, jobs=n_jobs, existing="refresh"
            )


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dandiset_id",
        default="000037",
        help=("ID of the dandiset from which to download assets"),
    )
    parser.add_argument(
        "--version",
        default=PUBLISHED_VERSION,
        help="version of the dandiset from which to download assets",
    )
    parser.add_argument(
        "--output", default=".", type=Path, help="where to store the dandiset files"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "does not download assets, just reports how many assets have "
            "been identified for download"
        ),
    )
    parser.add_argument(
        "--n_jobs", default=6, type=int, help="number of downloads to do in parallel"
    )

    # arguments applying only to dandiset 000037
    parser.add_argument(
        "--sess_ns",
        default="1-3",
        help="session numbers of assets to download (e.g., 1, 1-3 or all)",
    )
    parser.add_argument(
        "--mouse_ns",
        default="all",
        help="mouse numbers of assets to download (e.g., 1, 1-3 or all)",
    )
    parser.add_argument(
        "--excluded_sess",
        action="store_true",
        help=(
            "if True, all assets (even those excluded from the paper "
            "analyses) are downloaded."
        ),
    )
    parser.add_argument(
        "--mouse_df",
        default=DEFAULT_MOUSE_DF_PATH,
        type=Path,
        help="path to mouse_csv.df, if downloading by sess_ns and/or mouse_ns",
    )

    # type of asset to download
    parser.add_argument(
        "--incl_stim_templates",
        action="store_true",
        help=(
            "if True, assets containing the stimulus templates are "
            "downloaded (~1.5 to 3.1 GB per asset for dandiset 000037)"
        ),
    )
    parser.add_argument(
        "--incl_full_stacks",
        action="store_true",
        help="if True, assets containing the full imaging stack are downloaded",
    )

    args = parser.parse_args()

    download_dandiset_assets(**args.__dict__)
