from functools import reduce
import numpy as np

from .temporaldata import ArrayDict, IrregularTimeSeries, Interval, RegularTimeSeries


def concat(objs, sort=True):
    # check if all objects are of the same type
    obj_type = type(objs[0])
    if any(not isinstance(obj, obj_type) for obj in objs):
        raise ValueError(
            "All objects must be of the same type, got: {}".format(
                [type(obj) for obj in objs]
            )
        )

    if obj_type == IrregularTimeSeries:
        domain = reduce(lambda x, y: x | y, [obj.domain for obj in objs])

        keys = objs[0].keys()
        timekeys = objs[0].timekeys()
        for obj in objs:
            if set(obj.keys()) != set(keys):
                raise ValueError(
                    "All objects must have the same keys, got {} and {}".format(
                        keys, obj.keys()
                    )
                )
            if set(obj.timekeys()) != set(timekeys):
                raise ValueError(
                    "All objects must have the same timekeys, got {} and {}".format(
                        timekeys, obj.timekeys()
                    )
                )

        obj_concat_dict = {}
        for k in keys:
            obj_concat_dict[k] = np.concatenate([getattr(obj, k) for obj in objs])

        obj_concat = IrregularTimeSeries(
            **obj_concat_dict, timekeys=timekeys, domain=domain
        )

        if sort:
            obj_concat.sort()
    else:
        raise NotImplementedError(
            "Concatenation not implemented for type: {}".format(obj_type)
        )

    return obj_concat
