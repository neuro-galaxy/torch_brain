from brainsets.core import StringIntEnum


class Species(StringIntEnum):
    UNKNOWN = 0
    MACACA_MULATTA = 1
    HOMO_SAPIENS = HUMAN = 2
    MUS_MUSCULUS = 3
    MACACA_FASCICULARIS = NCBITaxon_9541 = 4
    CAENORHABDITIS_ELEGANS = 5


class Sex(StringIntEnum):
    r"""Follows the DANDI definition of sex.
    [Link](https://www.dandiarchive.org/handbook/135_validation/#missing-dandi-metadata)
    """

    UNKNOWN = U = 0
    MALE = M = 1
    FEMALE = F = 2
    OTHER = O = 3
