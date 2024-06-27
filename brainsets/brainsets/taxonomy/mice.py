from .core import StringIntEnum


class Cre_line(StringIntEnum):
    """
    ['Cux2-CreERT2',
    'Emx1-IRES-Cre',
    'Fezf2-CreER',
    'Nr5a1-Cre',
    'Ntsr1-Cre_GN220',
    'Pvalb-IRES-Cre',
    'Rbp4-Cre_KL100',
    'Rorb-IRES2-Cre',
    'Scnn1a-Tg3-Cre',
    'Slc17a7-IRES2-Cre',
    'Sst-IRES-Cre',
    'Tlx3-Cre_PL56',
    'Vip-IRES-Cre']"""

    CUX2_CREERT2 = 0
    EXM1_IRES_CRE = 1
    FEZF2_CREER = 2
    NR5A1_CRE = 3
    NTSR1_CRE_GN220 = 4
    PVALB_IRES_CRE = 5
    RBP4_CRE_KL100 = 6
    RORB_IRES2_CRE = 7
    SCNN1A_TG3_CRE = 8
    SLC17A7_IRES2_CRE = 9
    SST_IRES_CRE = 10
    TLX3_CRE_PL56 = 11
    VIP_IRES_CRE = 12


class Vis_areas(StringIntEnum):
    VIS_RL = 0  # Excluded
    VIS_PM = 1
    VIS_AL = 2
    VIS_AM = 3
    VIS_P = 4
    VIS_L = 5


class Depth_classes(StringIntEnum):
    # classes based on Allen Brain Observatory base: https://observatory.brain-map.org/visualcoding/
    DEPTH_CLASS_1 = 0  # 150-250 um
    DEPTH_CLASS_2 = 1  # 250-350 um
    DEPTH_CLASS_3 = 2  # 350-500 um
    DEPTH_CLASS_4 = 3  # 500-600 um
    DEPTH_CLASS_5 = 4  # 600+ um
