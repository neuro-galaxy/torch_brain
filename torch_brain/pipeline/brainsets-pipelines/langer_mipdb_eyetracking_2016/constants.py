"""Shared constants for the Langer MIPDB eyetracking 2016 pipeline.

Defines :
(:class:`Task`): task labels,
(:data:`PARADIGM_MAP`): annotation-code → paradigm name and task mappings ,
(:data:`PARADIGM_END_CODES`): naturalistic-viewing end codes,
(:data:`SAMPLES_COLUMNS`): SMI iView sample column renaming..

These constants are extracted from MISC_Readme files included in the dataset.

**Dataset documentation:**

    - Paper:
        Langer, N., Ho, E., Alexander, L. et al. A resource for assessing information processing
        in the developing brain using EEG and eye tracking. Sci Data 4, 170040 (2017).
        https://doi.org/10.1038/sdata.2017.40

    - Dataset portal:
        https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/index.html#
"""


class Task:
    RESTING_STATE = "RESTING_STATE"
    SEQUENCE_LEARNING = "SEQUENCE_LEARNING"
    VISUAL_SEARCH = "VISUAL_SEARCH"
    SURROUND_SUPPRESSION = "SURROUND_SUPPRESSION"
    CONTRAST_CHANGE_DETECTION = "CONTRAST_CHANGE_DETECTION"
    NATURALISTIC_VIEWING = "NATURALISTIC_VIEWING"


PARADIGM_MAP = {
    # annotation_code: (paradigm_name, task)
    90: ("Resting Paradigm", Task.RESTING_STATE),
    91: ("Sequence Learning Paradigm", Task.SEQUENCE_LEARNING),
    92: ("Symbol Search Paradigm", Task.VISUAL_SEARCH),
    93: ("Surround Suppression Paradigm Block 1", Task.SURROUND_SUPPRESSION),
    94: ("Contrast Change Paradigm Block 1", Task.CONTRAST_CHANGE_DETECTION),
    95: ("Contrast Change Paradigm Block 2", Task.CONTRAST_CHANGE_DETECTION),
    96: ("Contrast Change Paradigm Block 3", Task.CONTRAST_CHANGE_DETECTION),
    97: ("Surround Suppression Paradigm Block 2", Task.SURROUND_SUPPRESSION),
    81: ("Naturalistic Viewing Paradigm Video 1", Task.NATURALISTIC_VIEWING),
    82: ("Naturalistic Viewing Paradigm Video 2", Task.NATURALISTIC_VIEWING),
    83: ("Naturalistic Viewing Paradigm Video 3", Task.NATURALISTIC_VIEWING),
    84: ("Naturalistic Viewing Paradigm Video 4", Task.NATURALISTIC_VIEWING),
    85: ("Naturalistic Viewing Paradigm Video 5", Task.NATURALISTIC_VIEWING),
    86: ("Naturalistic Viewing Paradigm Video 6", Task.NATURALISTIC_VIEWING),
}

# Annotation codes that mark the end of a paradigm (by paradigm start code).
# Used only for NATURALISTIC_VIEWING paradigms (81-86). Other paradigms end at the
# last annotation before the next paradigm start (or last annotation in recording).
PARADIGM_END_CODES: dict[int, list[int]] = {
    81: [101, 0],
    82: [102, 0],
    83: [103, 0],
    84: [104, 0],
    85: [105, 0],
    86: [106, 0],
}

SAMPLES_COLUMNS = {
    "L Raw X [px]": "l_raw_x",
    "L Raw Y [px]": "l_raw_y",
    "R Raw X [px]": "r_raw_x",
    "R Raw Y [px]": "r_raw_y",
    "L Dia X [px]": "l_dia_x",
    "L Dia Y [px]": "l_dia_y",
    "R Dia X [px]": "r_dia_x",
    "R Dia Y [px]": "r_dia_y",
    "L CR1 X [px]": "l_cr1_x",
    "L CR1 Y [px]": "l_cr1_y",
    "L CR2 X [px]": "l_cr2_x",
    "L CR2 Y [px]": "l_cr2_y",
    "R CR1 X [px]": "r_cr1_x",
    "R CR1 Y [px]": "r_cr1_y",
    "R CR2 X [px]": "r_cr2_x",
    "R CR2 Y [px]": "r_cr2_y",
    "L POR X [px]": "l_por_x",
    "L POR Y [px]": "l_por_y",
    "R POR X [px]": "r_por_x",
    "R POR Y [px]": "r_por_y",
    "latency": "latency",
    "L Validity": "l_validity",
    "R Validity": "r_validity",
    "Pupil Confidence": "pupil_confidence",
    "L Plane": "l_plane",
    "R Plane": "r_plane",
    "H POS X [mm]": "h_pos_x",
    "H POS Y [mm]": "h_pos_y",
    "H POS Z [mm]": "h_pos_z",
    "H ROT X [°]": "h_rot_x",
    "H ROT Y [°]": "h_rot_y",
    "H ROT Z [°]": "h_rot_z",
    "L EPOS X": "l_epos_x",
    "L EPOS Y": "l_epos_y",
    "L EPOS Z": "l_epos_z",
    "R EPOS X": "r_epos_x",
    "R EPOS Y": "r_epos_y",
    "R EPOS Z": "r_epos_z",
    "L GVEC X": "l_gvec_x",
    "L GVEC Y": "l_gvec_y",
    "L GVEC Z": "l_gvec_z",
    "R GVEC X": "r_gvec_x",
    "R GVEC Y": "r_gvec_y",
    "R GVEC Z": "r_gvec_z",
}
