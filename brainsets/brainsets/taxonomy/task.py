from .core import StringIntEnum


class Task(StringIntEnum):
    REACHING = 0

    # For datasets where no tasks are involved
    FREE_BEHAVIOR = 1

    # A Shenoy-style task involving handwriting different characters.
    DISCRETE_WRITING_CHARACTER = 2

    DISCRETE_WRITING_LINE = 3

    CONTINUOUS_WRITING = 4

    # Allen data
    DISCRETE_VISUAL_CODING = 5

    # speech
    DISCRETE_SPEAKING_CVSYLLABLE = 6

    # Full sentence speaking
    CONTINUOUS_SPEAKING_SENTENCE = 7


class REACHING(StringIntEnum, parent=Task.REACHING):
    """A classic BCI task involving reaching to a 2d target."""

    RANDOM = 0
    HOLD = 1
    REACH = 2
    RETURN = 3
    INVALID = 4
    OUTLIER = 5
    MAX = 6


class Stimulus(StringIntEnum):
    """Stimuli can variously act like inputs (for conditioning) or like outputs."""

    TARGET2D = 0
    TARGETON = 1
    GO_CUE = 2
    TARGETACQ = 3

    DRIFTING_GRATINGS = 4
