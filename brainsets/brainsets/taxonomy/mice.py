from brainsets.core import StringIntEnum


class Cre_line(StringIntEnum):
    CUX2_CREERT2 = 0
    EMX1_IRES_CRE = 1
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


class BrainRegion(StringIntEnum):
    VIS_RL = VISRL = 0  # Excluded
    VIS_PM = VISPM = 1
    VIS_AL = VISAL = 2
    VIS_AM = VISAM = 3
    VIS_P = VISP = 4
    VIS_L = VISL = 5
