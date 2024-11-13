from functools import reduce
import numpy as np

from .temporaldata import ArrayDict, IrregularTimeSeries, Interval, RegularTimeSeries


def concat(objs, sort=True):
    """Concatenates multiple time series objects into a single object.

    Args:
        objs (List[Union[IrregularTimeSeries, RegularTimeSeries]]): List of time series objects to concatenate.
        sort (bool, optional): Whether to sort the resulting time series by timestamps. Only applies to IrregularTimeSeries. Defaults to True.

    Returns:
        Union[IrregularTimeSeries, RegularTimeSeries]: The concatenated time series object.

    Raises:
        ValueError: If objects are not all of the same type or don't have matching keys.
        NotImplementedError: If concatenation is not implemented for the given object type.

    .. code-block:: python

        import numpy as np
        from temporaldata import IrregularTimeSeries, concat

        ts1 = IrregularTimeSeries(
            timestamps=np.array([0.0, 1.0]),
            values=np.array([1.0, 2.0])
        )
        ts2 = IrregularTimeSeries(
            timestamps=np.array([2.0, 3.0]),
            values=np.array([3.0, 4.0])
        )

        ts_concat = concat([ts1, ts2])
        >>> IrregularTimeSeries(
            timestamps=[4],
            values=[4]
        )
    """
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
