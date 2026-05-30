"""Script adapted from this notebook: 
https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
"""

import argparse
import requests
import os
import shutil

from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


if __name__ == "__main__":
    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./raw", help="Output directory"
    )
    parser.add_argument("--session_id", type=int, default=None)
    parser.add_argument(
        "--download_lfp", type=bool, default=False, help="Download LFP data"
    )
    args = parser.parse_args()

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # download data for session
    truncated_file = True
    directory = os.path.join(args.output_dir + "/session_" + str(args.session_id))

    while truncated_file:
        session = cache.get_session_data(args.session_id)
        try:
            print(session.specimen_name)
            truncated_file = False
        except OSError:
            shutil.rmtree(directory)
            print(" Truncated spikes file, re-downloading")

    if args.download_lfp:
        for probe_id, probe in session.probes.iterrows():
            print(" " + probe.description)
            truncated_lfp = True

            while truncated_lfp:
                try:
                    lfp = session.get_lfp(probe_id)
                    truncated_lfp = False
                except OSError:
                    fname = directory + "/probe_" + str(probe_id) + "_lfp.nwb"
                    os.remove(fname)
                    print("  Truncated LFP file, re-downloading")
                except ValueError:
                    print("  LFP file not found.")
                    truncated_lfp = False
