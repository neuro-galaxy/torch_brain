from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
import numpy as np
import allensdk.brain_observatory.stimulus_info as stim_info
import pandas as pd
from allensdk.brain_observatory.brain_observatory_exceptions import (
    EpochSeparationException,
)
import matplotlib.pyplot as plt
from simplejson.errors import JSONDecodeError


if __name__ == "__main__":
    # boc = BrainObservatoryCache(manifest_file='/home/mila/x/xuejing.pan/scratch/manifest.json')
    boc = BrainObservatoryCache(
        manifest_file="/network/projects/neuro-galaxy/data/raw/allen_brain_observatory_calcium/manifest.json"
    )
    all_dg_exps = boc.get_ophys_experiments(stimuli=[stim_info.DRIFTING_GRATINGS])
    num_exps = len(all_dg_exps)

    ##CREATING NEW CSV FILES
    columns = [
        "exp_id",
        "subject_id",
        "cre_line",
        "depth",
        "num_seqs",
        "num_ROIs",
        "num_timepoints",
    ]

    collected_data = []

    for i in range(num_exps):
        print("AT FILE: ", i)

        exp = all_dg_exps[i]
        session_id = all_dg_exps[i]["id"]
        try:
            exp = boc.get_ophys_experiment_data(exp["id"])
        except OSError as e:
            continue
        except JSONDecodeError as e:
            continue
        exp_cre_line = all_dg_exps[i]["cre_line"]
        exp_depth = all_dg_exps[i]["imaging_depth"]
        subject_id = all_dg_exps[i]["donor_name"]

        print(session_id)

        traces = exp.get_dff_traces()
        num_rois = traces[1].shape[0]

        num_timepoints = 0

        try:
            master_stim_table = exp.get_stimulus_epoch_table()
            for i, stim in enumerate(master_stim_table["stimulus"]):
                if stim == "drifting_gratings":
                    curr_time_points = (
                        master_stim_table["end"][i] - master_stim_table["start"][i]
                    )
                    num_timepoints += curr_time_points

        except EpochSeparationException as e:
            # num_timepoints = 0
            continue

        stim_table = exp.get_stimulus_table("drifting_gratings")

        num_seqs = stim_table.index[-1] + 1

        new_row = {
            "exp_id": session_id,
            "subject_id": subject_id,
            "cre_line": exp_cre_line,
            "depth": exp_depth,
            "num_seqs": num_seqs,
            "num_ROIs": num_rois,
            "num_timepoints": num_timepoints,
        }
        print(new_row)

        collected_data.append(new_row)
        print("TOTAL FILE: ", num_exps)

    df = pd.DataFrame(collected_data)
    df.to_csv(
        "/home/mila/x/xuejing.pan/POYO/project-kirby/kirby/AllenBOmeta.csv", index=False
    )
