import json
import numpy as np
import sys
import argparse
import datetime
import logging

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)
from brainsets_utils import DatasetBuilder, find_files_by_extension
from brainsets_utils.taxonomy import Species, Task


def extract_labeled(data):
    labeldict = {}
    # Used units is the number units tracked in the neural data
    used_units = len(data["trace_array"])
    for i in range(1, used_units + 1):
        labeldict[str(i)] = {
            "label": "missing",
            "neuron_class": "missing",
            "LR": "missing",
            "region": "missing",
            "roi_id": [-1],
            "confidence": -1,
            "DV": "missing",
        }
    for i in range(1, used_units + 1):
        if str(i) in data["labeled"]:
            labeldict[str(i)] = data["labeled"][str(i)]
    label = []
    neuron_class = []
    LR = []
    region = []
    roi_id = []
    confidence = []
    DV = []
    for i in range(1, used_units + 1):
        label.append(labeldict[str(i)]["label"])
        neuron_class.append(labeldict[str(i)]["neuron_class"])
        LR.append(labeldict[str(i)]["LR"])
        if labeldict[str(i)]["region"] is None:
            region.append("null")
        else:
            region.append(labeldict[str(i)]["region"])
        roi_id.append(labeldict[str(i)]["roi_id"])
        confidence.append(labeldict[str(i)]["confidence"])
        DV.append(labeldict[str(i)]["DV"])

    unitdata = ArrayDict(
        id=np.array([int(x) for x in labeldict.keys()], dtype=int),
        label=np.array(label),
        neuron_class=np.array(neuron_class),
        LR=np.array(LR),
        region=np.array(region),
        confidence=np.array(confidence),
        DV=np.array(DV),
    )

    return unitdata


def extract_neural_data(data):
    # transposing because trace_array is in the shape(n, t) and RegularTimeSeries needs the shape(t,n)
    trace_array = np.array(data["trace_array"]).transpose(1, 0)
    # trace_original is in the shape (t,n) already so no transpose is needed.
    trace_original = np.array(data["trace_original"])
    avg_timestep = data["avg_timestep"] * 60
    end = (len(trace_original) - 1) * avg_timestep
    neural_data = RegularTimeSeries(
        sampling_rate=1 / avg_timestep,
        trace_array=trace_array,
        trace_original=trace_original,
        domain=Interval(0, end),
    )
    return neural_data


def extract_dataset_info(data):
    dataset_info = data["dataset_type"]
    neuropal = "neuropal" in dataset_info
    dataset_type = dataset_info[0]

    return dataset_type, neuropal


def extract_behaviour_data(data):
    velocity = np.array(data["velocity"])
    head_curvature = np.array(data["head_curvature"])
    pumping = np.array(data["pumping"])
    angular_velocity = np.array(data["angular_velocity"])
    body_curvature = np.array(data["body_curvature"])
    reversal_events = data["reversal_events"]
    reversal_events_list = [False for i in range(len(velocity))]
    for i in range(len(velocity)):
        for reversal_event in reversal_events:
            if i >= reversal_event[0] and i <= reversal_event[1]:
                reversal_events_list[i] = True
                break
    avg_timestep = data["avg_timestep"] * 60
    end = (len(angular_velocity) - 1) * avg_timestep
    reversal_events_list = np.array(reversal_events_list)
    behaviour = RegularTimeSeries(
        sampling_rate=1 / avg_timestep,
        velocity=velocity,
        head_curvature=head_curvature,
        pumping=pumping,
        angular_velocity=angular_velocity,
        body_curvature=body_curvature,
        reversal_events=reversal_events_list,
        domain=Interval(0, end),
    )
    return behaviour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="wormwideweb",
        origin_version="unknown",
        derived_version="1.0.0",
        source="https://www.wormwideweb.org/dataset.html",
        description="This dataset contains dozens of freely behaving C. elegans whole-brain sessions"
        " The sessions track the behaviour and neural traces along with other data such as",
    )

    for file_path in find_files_by_extension(db.raw_folder_path, ".json"):

        # input dir might contain some metadata files
        if file_path.endswith("summary.json"):
            continue

        with db.new_session() as session:
            logging.info(f"Processing file: {file_path}")

            year = int(file_path[-18:-14])
            month = int(file_path[-13:-11])
            day = int(file_path[-10:-8])
            date = datetime.datetime(year, month, day)

            id = file_path[-18:-5]
            subject_id = id + "_subject"

            session.register_subject(
                id=subject_id, species=Species.CAENORHABDITIS_ELEGANS
            )

            session.register_session(
                id=id,
                recording_date=date,
                task=Task.FREE_BEHAVIOR,
            )

            with open(file_path, "r") as f:
                jsondata = json.load(f)

            units = extract_labeled(jsondata)
            session.register_sortset(
                id=id,
                units=units,
            )

            neural_data = extract_neural_data(jsondata)
            behaviour = extract_behaviour_data(jsondata)
            dataset_type, neuropal = extract_dataset_info(jsondata)

            data = Data(
                dataset_type=dataset_type,
                neuropal=neuropal,
                units=units,
                neural_data=neural_data,
                angular_velocity=behaviour,
                domain=(neural_data.domain | behaviour.domain),
            )

            session.register_data(data)
            session.save_to_disk()

    db.finish()


if __name__ == "__main__":
    main()
