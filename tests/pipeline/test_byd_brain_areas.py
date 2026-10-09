from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _utils import add_pipelines_to_path

from torch_brain.datasets.KelesBYD2024 import KelesBYD2024

add_pipelines_to_path()
from keles_byd_2024 import pipeline as byd_pipeline  # noqa: E402


def _write_brain_area_labels(root: Path, text: str) -> None:
    brain_areas = root / "brain_areas"
    brain_areas.mkdir(parents=True, exist_ok=True)
    (brain_areas / "brain_area_labels.csv").write_text(text)


def test_build_channels_adds_byd_brain_area_labels_with_coordinate_key(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
            "ds004798,sub-p41cs,LACC1,0.18,26.56,17.92,"
            "Left ACC,shaft,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-rostralanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
            "ds004798,sub-p41cs,RAMY1,21.12,-5.97,-14.22,"
            "Right amygdala,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "Right-Amygdala,Right-Amygdala\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1", "RAMY1", "LACC1"],
            "x": [0.18, 21.12, -4.36],
            "y": [26.56, -5.97, 28.32],
            "z": [17.92, -14.22, 19.36],
        }
    )

    channels = byd_pipeline._build_channels(electrode_df, subject_number=41)

    assert channels.name.tolist() == ["LACC1", "RAMY1", "LACC1"]
    assert channels.label_dkt.tolist() == [
        "ctx-lh-rostralanteriorcingulate",
        "Right-Amygdala",
        "ctx-lh-caudalanteriorcingulate",
    ]
    assert channels.provided_location.tolist() == [
        "Left ACC",
        "Right amygdala",
        "Left ACC",
    ]


def test_build_channels_allows_duplicate_session_rows_with_same_labels(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,session,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,P41CSR1,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
            "ds004798,sub-p41cs,P41CSR2,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1"],
            "x": [-4.36],
            "y": [28.32],
            "z": [19.36],
        }
    )

    channels = byd_pipeline._build_channels(electrode_df, subject_number=41)

    assert channels.label_dkt.tolist() == ["ctx-lh-caudalanteriorcingulate"]


def test_build_channels_rejects_missing_byd_brain_area_labels(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1"],
            "x": [0.18],
            "y": [26.56],
            "z": [17.92],
        }
    )

    with pytest.raises(ValueError, match="Missing brain area labels.*LACC1"):
        byd_pipeline._build_channels(electrode_df, subject_number=41)


def test_build_channels_allows_unlabelable_byd_contacts_without_coordinates(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1", "LSPE1"],
            "location": ["Left ACC", "Left superior parietal"],
            "x": [-4.36, np.nan],
            "y": [28.32, np.nan],
            "z": [19.36, np.nan],
        }
    )

    channels = byd_pipeline._build_channels(electrode_df, subject_number=41)

    assert channels.label_dkt.tolist() == ["ctx-lh-caudalanteriorcingulate", ""]
    assert channels.label_destrieux.tolist() == [
        "ctx_lh_G_and_S_cingul-Ant",
        "",
    ]
    assert channels.provided_location.tolist() == [
        "Left ACC",
        "Left superior parietal",
    ]
    assert channels.coordinate_assumption.tolist() == [
        "released_nwb_xyz_as_mni_template_ras",
        "nwb_xyz_missing",
    ]
    assert channels.label_status.tolist() == [
        "exploratory_template_space",
        "unlabeled_missing_coordinates",
    ]
    np.testing.assert_allclose(
        channels.coord_byd_mni152_r, [-4.36, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        channels.coord_byd_mni152_a, [28.32, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        channels.coord_byd_mni152_s, [19.36, np.nan], equal_nan=True
    )


def test_build_channels_treats_blank_byd_coordinates_as_unlabelable(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1", "LSPE1"],
            "location": ["Left ACC", "Left superior parietal"],
            "x": [-4.36, ""],
            "y": [28.32, ""],
            "z": [19.36, ""],
        }
    )

    channels = byd_pipeline._build_channels(electrode_df, subject_number=41)

    assert channels.label_dkt.tolist() == ["ctx-lh-caudalanteriorcingulate", ""]
    assert channels.label_status.tolist() == [
        "exploratory_template_space",
        "unlabeled_missing_coordinates",
    ]


def test_build_channels_serializes_nonfinite_byd_coordinate_aliases_as_nan(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1", "LSPE1"],
            "location": ["Left ACC", "Left superior parietal"],
            "x": [-4.36, np.inf],
            "y": [28.32, -np.inf],
            "z": [19.36, np.inf],
        }
    )

    channels = byd_pipeline._build_channels(electrode_df, subject_number=41)

    assert channels.label_status.tolist() == [
        "exploratory_template_space",
        "unlabeled_missing_coordinates",
    ]
    assert channels.qc_status.tolist() == [
        "not_reviewed",
        "not_labelable_missing_coordinates",
    ]
    np.testing.assert_allclose(
        channels.coord_byd_mni152_r, [-4.36, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        channels.coord_byd_mni152_a, [28.32, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        channels.coord_byd_mni152_s, [19.36, np.nan], equal_nan=True
    )


def test_build_channels_rejects_uncovered_byd_subject(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", tmp_path)
    _write_brain_area_labels(
        tmp_path,
        (
            "dataset_id,subject,electrode,mni_x,mni_y,mni_z,"
            "provided_location,trajectory_role,coordinate_assumption,"
            "label_status,qc_status,label_dkt,label_destrieux\n"
            "ds004798,sub-p41cs,LACC1,-4.36,28.32,19.36,"
            "Left ACC,deep_target,released_nwb_xyz_as_mni_template_ras,"
            "exploratory_template_space,not_reviewed,"
            "ctx-lh-caudalanteriorcingulate,ctx_lh_G_and_S_cingul-Ant\n"
        ),
    )
    electrode_df = pd.DataFrame(
        {
            "origchannel_name": ["LACC1"],
            "x": [-4.36],
            "y": [28.32],
            "z": [19.36],
        }
    )

    with pytest.raises(ValueError, match="No brain area labels.*subject 49"):
        byd_pipeline._build_channels(electrode_df, subject_number=49)


def test_get_channel_metadata_includes_optional_byd_brain_area_labels(monkeypatch):
    ds = KelesBYD2024.__new__(KelesBYD2024)

    channels = SimpleNamespace(
        id=np.array(["0", "1"], dtype=str),
        name=np.array(["LACC1", "RAMY1"], dtype=str),
        included=np.array([True, False], dtype=bool),
        coord_byd_mni152_r=np.array([0.18, 21.12], dtype=float),
        coord_byd_mni152_a=np.array([26.56, -5.97], dtype=float),
        coord_byd_mni152_s=np.array([17.92, -14.22], dtype=float),
        label_dkt=np.array(
            ["ctx-lh-rostralanteriorcingulate", "Right-Amygdala"], dtype=object
        ),
        label_destrieux=np.array(
            ["ctx_lh_G_and_S_cingul-Ant", "Right-Amygdala"], dtype=object
        ),
        provided_location=np.array(["Left ACC", "Right amygdala"], dtype=object),
        trajectory_role=np.array(["shaft", "deep_target"], dtype=object),
        coordinate_assumption=np.array(
            [
                "released_nwb_xyz_as_mni_template_ras",
                "released_nwb_xyz_as_mni_template_ras",
            ],
            dtype=object,
        ),
        label_status=np.array(["exploratory_template_space"] * 2, dtype=object),
        qc_status=np.array(["not_reviewed"] * 2, dtype=object),
    )
    monkeypatch.setattr(
        ds,
        "get_recording",
        lambda _recording_id: SimpleNamespace(channels=channels),
    )

    arrays = ds.get_channel_metadata("sub-CS41_ses-P41CSR1")

    assert "coords" not in arrays
    assert "coords_type" not in arrays
    np.testing.assert_allclose(
        arrays["coordinate_frames"]["byd_mni152_ras"],
        np.array([[0.18, 26.56, 17.92], [21.12, -5.97, -14.22]]),
    )
    assert arrays["label_dkt"].tolist() == [
        "ctx-lh-rostralanteriorcingulate",
        "Right-Amygdala",
    ]
    assert arrays["label_destrieux"].tolist() == [
        "ctx_lh_G_and_S_cingul-Ant",
        "Right-Amygdala",
    ]
    assert arrays["provided_location"].tolist() == ["Left ACC", "Right amygdala"]
    assert arrays["trajectory_role"].tolist() == ["shaft", "deep_target"]
    assert arrays["coordinate_assumption"].tolist() == [
        "released_nwb_xyz_as_mni_template_ras",
        "released_nwb_xyz_as_mni_template_ras",
    ]
    assert arrays["label_status"].tolist() == ["exploratory_template_space"] * 2
    assert arrays["qc_status"].tolist() == ["not_reviewed"] * 2


def test_get_channel_metadata_rejects_optional_byd_field_length_mismatch(monkeypatch):
    ds = KelesBYD2024.__new__(KelesBYD2024)
    channels = SimpleNamespace(
        id=np.array(["0", "1"]),
        name=np.array(["LACC1", "RAMY1"]),
        included=np.array([True, False]),
        coord_byd_mni152_r=np.array([0.18, 21.12]),
        coord_byd_mni152_a=np.array([26.56, -5.97]),
        coord_byd_mni152_s=np.array([17.92, -14.22]),
        label_status=np.array(["exploratory_template_space"]),
    )
    monkeypatch.setattr(
        ds,
        "get_recording",
        lambda _recording_id: SimpleNamespace(channels=channels),
    )

    with pytest.raises(ValueError, match="label_status length mismatch"):
        ds.get_channel_metadata("sub-CS41_ses-P41CSR1")
