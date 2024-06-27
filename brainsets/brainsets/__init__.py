__version__ = "0.1.0"

from .dir_utils import find_files_by_extension, make_directory

from . import taxonomy
from . import processing

from . import mat_utils
from . import dandi_utils

import datetime


def string_int_enum_serialize_fn(obj, serialize_fn_map=None):
    return str(obj)


def datetime_serialize_fn(obj, serialize_fn_map=None):
    return str(obj)


serialize_fn_map = {
    taxonomy.StringIntEnum: string_int_enum_serialize_fn,
    datetime.datetime: datetime_serialize_fn,
}
