import argparse
import datetime
import logging
from typing import Tuple
import numpy as np
import os
import tqdm
import re

from brainsets_utils.taxonomy import speech

import pandas as pd
import scipy.io as sio

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.taxonomy import (
    RecordingTech,
    Task,
    SubjectDescription,
    Sex,
    Species,
)

logging.basicConfig(level=logging.INFO)

# 20 ms bins
FREQ = 50

vocab = "abcdefghijklmnopqrstuvwxyz'- "


def clean(sentence):
    sentence = "".join(
        [
            x
            for x in sentence.lower().strip().replace("--", "")
            if x == " " or x in vocab
        ]
    )
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def spike_array_to_timestamps(
    arr: np.ndarray, freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a matrix corresponding to a list of threshold crossings into a list
    of spike times and spike ids.
    """
    spike_timestamps = []
    spike_ids = []

    while (arr > 0).any():
        idx_time, idx_spike = np.where(arr > 0)
        spike_timestamps.append(idx_time / freq)
        spike_ids.append(idx_spike)
        arr = (arr - 1) * (arr > 0)

    spike_timestamps = np.concatenate(spike_timestamps).squeeze()
    spike_ids = np.concatenate(spike_ids).squeeze().astype(np.int32)

    idx = np.argsort(spike_timestamps)
    spike_timestamps, spike_ids = spike_timestamps[idx], spike_ids[idx]

    return spike_timestamps, spike_ids


"""
Layout according to paper

											   ^
											   |
											   |
											Superior
											
				Area 44 Superior 					Area 6v Superior
				192 193 208 216 160 165 178 185     062 051 043 035 094 087 079 078 
				194 195 209 217 162 167 180 184     060 053 041 033 095 086 077 076 
				196 197 211 218 164 170 177 189     063 054 047 044 093 084 075 074 
				198 199 210 219 166 174 173 187     058 055 048 040 092 085 073 072 
				200 201 213 220 168 176 183 186     059 045 046 038 091 082 071 070 
				202 203 212 221 172 175 182 191     061 049 042 036 090 083 069 068 
				204 205 214 223 161 169 181 188     056 052 039 034 089 081 067 066 
				206 207 215 222 163 171 179 190     057 050 037 032 088 080 065 064 
<-- Anterior 															Posterior -->
				Area 44 Inferior 					Area 6v Inferior 
				129 144 150 158 224 232 239 255     125 126 112 103 031 028 011 008 
				128 142 152 145 226 233 242 241     123 124 110 102 029 026 009 005 
				130 135 148 149 225 234 244 243     121 122 109 101 027 019 018 004 
				131 138 141 151 227 235 246 245     119 120 108 100 025 015 012 006 
				134 140 143 153 228 236 248 247     117 118 107 099 023 013 010 003 
				132 146 147 155 229 237 250 249     115 116 106 097 021 020 007 002 
				133 137 154 157 230 238 252 251     113 114 105 098 017 024 014 000 
				136 139 156 159 231 240 254 253     127 111 104 096 030 022 016 001 
				
										    Inferior
											   |
											   |
											   ∨
"""


def get_unit_metadata():
    recording_tech = RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS
    unit_meta = []
    for i in range(256):
        unit_id = f"group_0/elec{i:03d}/multiunit_0"
        unit_meta.append(
            {"id": unit_id, "unit_number": i, "count": -1, "type": int(recording_tech)}
        )
    unit_meta_df = pd.DataFrame(unit_meta)
    units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

    return units


def stack_trials(mat_data):
    """Stack all the trial data into a single array."""
    threshold_xings = mat_data["tx1"].squeeze().tolist()
    threshold_xings = [x[:, :256] for x in threshold_xings]
    trial_times = [x.shape[0] / FREQ for x in threshold_xings]
    trial_bounds = np.cumsum(np.concatenate([[0], trial_times]))

    threshold_xings = [spike_array_to_timestamps(x, FREQ) for x in threshold_xings]

    # Create the corresponding intervals for that dataset
    trials = Interval(start=trial_bounds[:-1], end=trial_bounds[1:])

    units = get_unit_metadata()

    timestamps = np.concatenate(
        [x + trial_bounds[i] for i, (x, _) in enumerate(threshold_xings)]
    )
    unit_index = np.concatenate([y for _, y in threshold_xings])

    spikes = IrregularTimeSeries(
        timestamps=np.array(timestamps),
        unit_index=np.array(unit_index),
        domain=trials,
    )

    sentences = [clean(x) for x in mat_data["sentenceText"].squeeze().tolist()]

    # Write things out phonetically, which is the actual target.
    phonemes = [
        speech.to_phonemes(x) for x in mat_data["sentenceText"].squeeze().tolist()
    ]
    cypher = [x for x, _ in phonemes]

    # We have to pad phonemes in order to store the data in a regular array.
    phonemes = [x for _, x in phonemes]
    phoneme_len = [len(x) for x in phonemes]
    max_len = max(phoneme_len)
    phonemes = [
        np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=0)
        for x in phonemes
    ]

    # We assign the labels to the start of each trial.
    sentences = IrregularTimeSeries(
        timestamps=trial_bounds[:-1],
        block_num=mat_data["blockIdx"].squeeze(),
        transcript=np.array(sentences),
        phonemes=np.array(phonemes),
        phonemes_len=np.array(phoneme_len),
        phonemes_cypher=np.array(cypher),
        domain="auto",
    )

    return spikes, units, trials, sentences


def get_subject():
    return SubjectDescription(
        id="T12",
        species=Species.from_string("HUMAN"),
        sex=Sex.from_string("M"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        experiment_name="willett_henderson_speech_2023",
        origin_version="1.0.0",
        derived_version="1.0.0",
        source="https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq",
        description="This dataset contains neural activity and speech stimulus data from a paralyzed human participant who was implanted with two 96-electrode intracortical arrays in the hand/arm area of primary motor cortex. The participant attempted to read sentences aloud, and a deep learning model decoded intended speech from the neural activity.",
    )

    # Each of these corresponds to a train dataset
    subject = get_subject()  # Participant pseudonym
    for file_path in tqdm.tqdm(
        find_files_by_extension(os.path.join(args.input_dir, "train"), ".mat")
    ):
        with db.new_session() as session:
            load_keys = ["sentenceText", "tx1", "blockIdx"]
            mat_data = sio.loadmat(file_path, variable_names=load_keys)
            num_train_trials = mat_data["tx1"].shape[1]

            # Also concatenate the data from the validation split. Note the validation
            # split is actually called test here.
            valid_path = os.path.join(args.input_dir, "test", file_path.split("/")[-1])
            valid_mat_data = sio.loadmat(valid_path, variable_names=load_keys)
            num_valid_trials = valid_mat_data["tx1"].shape[1]

            for key in load_keys:
                mat_data[key] = np.concatenate(
                    [mat_data[key].squeeze(), valid_mat_data[key].squeeze()], axis=0
                )

            session.register_subject(subject)

            # Sortset ID and session ID should be the same here.
            session_id = os.path.splitext(os.path.basename(file_path))[0]
            _, year, month, day = session_id.split(".")
            recording_date = datetime.datetime(int(year), int(month), int(day))
            session.register_session(
                id=session_id,
                recording_date=recording_date,
                task=Task.CONTINUOUS_SPEAKING_SENTENCE,
            )

            spikes, units, trials, sentences = stack_trials(mat_data)

            # Concatenate the data from both cases
            session.register_sortset(
                id=session_id,
                units=units,
            )

            data = Data(
                spikes=spikes,
                units=units,
                trials=trials,
                speech=sentences,
                domain=spikes.domain,
            )

            train_mask = np.zeros(num_train_trials + num_valid_trials, dtype=bool)
            train_mask[:num_train_trials] = True

            train_split = trials.select_by_mask(train_mask)
            valid_split = trials.select_by_mask(~train_mask)

            session.register_data(data)
            session.register_split("train", train_split)
            session.register_split("valid", valid_split)
            session.save_to_disk()

    db.finish()


if __name__ == "__main__":
    main()
