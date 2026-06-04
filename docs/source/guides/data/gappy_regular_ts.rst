.. currentmodule:: torch_brain.data

Regular Time Series with Gaps
=============================

Some signals are *almost* regular: they are sampled at a fixed rate, but have
missing samples or chunks of samples. Behavioral streams in neuroscience
experiments are a common example: a sensor briefly disconnects or a chunk of
data is lost between recording segments.

While such signals can be stored as :obj:`IrregularTimeSeries`, there is a
certain benefit to storing signals as :obj:`RegularTimeSeries`: slicing
precision. A :obj:`RegularTimeSeries` slice always returns the same number of
points for the same window width.

This motivated us to extend the interface of :obj:`RegularTimeSeries` to
support *gappy* regular time series, which keeps that reliable slicing while
allowing for missing time points. The main idea, simply, is to represent the
missing timestamps with NaNs, while explicitly tracking which samples are real
and which are gap-fill.

.. dropdown:: More on slicing precision
    :icon: light-bulb
    :color: success

    Slicing an :obj:`IrregularTimeSeries` close to real timestamps can return
    :math:`N` or :math:`N-1` points depending on floating-point rounding
    errors. So, in practice, windowed sampling, *effectively*, behaves
    non-deterministically. More precisely, this happens because we store
    timestamps of irregular time series in floating point format
    (:obj:`numpy.float64`). Slicing involves a search in this floating point
    space, and comparisons between floating numbers are notoriously unreliable.

    A :obj:`RegularTimeSeries` internally represents time as *integer indices*,
    where it is easier to control all the messy floating point numerics. As a
    result, a slice always returns the same number of points for the same
    window width.

Creating a gappy series
-----------------------

Use :meth:`RegularTimeSeries.from_gappy_timeseries` when you have
regularly-sampled but gappy timestamps and value arrays. Each sample is snapped
to a regular grid at ``sampling_rate``, and missing samples are filled with
a configurable gap value.

.. code-block:: pycon

    >>> from torch_brain.data import RegularTimeSeries

    >>> # Signal sampled at 1 Hz but a few samples dropped: t = 3, 6, 7
    >>> ts = [0., 1., 2., 4., 5., 8., 9.,]
    >>> values = [0.1, 0.4, 0.2, 0.1, 0.0, 0.3, 0.5,]

    >>> signal = RegularTimeSeries.from_gappy_timeseries(
    ...     timestamps=ts,
    ...     values=values,
    ...     sampling_rate=1.0,
    ... )

    >>> len(signal)
    10

    >>> signal.timestamps
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

    >>> signal.values
    array([0.1, 0.4, 0.2, nan, 0.1, 0.0, nan, nan, 0.3, 0.5])

The resulting object behaves like any other :obj:`RegularTimeSeries`, just with
some gap-fill values.

.. tip::

    You can customize which gap-fill values are used for
    different data types. To do this, see the ``gap_value`` parameter of
    :meth:`~RegularTimeSeries.from_gappy_timeseries`.


Domain
------

While a contiguous :obj:`RegularTimeSeries` has a contiguous
:attr:`~RegularTimeSeries.domain`, a gappy series carries a non-contiguous
domain that excludes the gap regions. For the example above:

.. code-block:: pycon

   >>> signal.domain.start, signal.domain.end
   array([0., 4., 8.]), array([3., 6., 10.])

This is :math:`[0, 3) \cup [4, 6) \cup [8, 10)`.


Identifying real vs. gap-fill samples
-------------------------------------

To help you decipher which samples are *real* and which are *gap-fills*, we
provide the :meth:`~RegularTimeSeries.index_mask` method. It returns a
boolean mask marking which positions hold real observations:

.. code-block:: pycon

   >>> signal.index_mask()
   array([ True,  True,  True, False,  True,  True, False, False,  True,  True])

   >>> # to get back "real" signal values:
   >>> signal.values[signal.index_mask()]
   array([0.1, 0.4, 0.2, 0.1, 0.0, 0.3, 0.5])

For a contiguous series, :meth:`~RegularTimeSeries.index_mask` returns an
all-``True`` array.


:meth:`~RegularTimeSeries.is_gappy` is another convenient introspection method:

.. code-block:: pycon

   >>> signal.is_gappy()
   True

   >>> contiguous = RegularTimeSeries(values=[0.1, 0.4, 0.2], sampling_rate=1.0)
   >>> contiguous.is_gappy()
   False


Slicing
-------

Slicing mostly follows the normal :obj:`RegularTimeSeries` semantics, with two
additions specific to gappy series:

- **Edge gaps are trimmed.** If a slice boundary falls inside a gap, the
  returned arrays will not begin or end with gap-fill samples. That is, slicing
  always returns data bracketed by real samples.
- **Internal gaps are preserved if needed.** Gap-fill samples in the middle of the
  requested window remain in place; the returned object is itself gappy.

.. code-block:: pycon

    >>> sliced = signal.slice(3.0, 9.0, reset_origin=False)
    >>> sliced.timestamps
    array([ 4.,  5.,  6.,  7.,  8.])
    >>> sliced.values
    array([0.1, 0. , nan, nan, 0.3])
    >>> sliced.domain.start, sliced.domain.end
    array([4., 8.]), array([6., 9.])

Notice that the domain does *not* start at :math:`t = 3`, and
the gap between :math:`t = 6` and :math:`t = 8` is preserved.

.. figure:: /_static/gappy-slice.png
   :width: 65%
   :align: center

   Slicing gappy RegularTimeSeries

A slice that is entirely within a contiguous section is no longer gappy:

.. code-block:: pycon

   >>> sliced = signal.slice(0.0, 2.0, reset_origin=False)
   >>> sliced.timestamps
   array([ 0.,  1.])
   >>> sliced.is_gappy()
   False

A slice that falls entirely within a gap returns an empty series:

.. code-block:: pycon

   >>> empty = signal.slice(6.0, 8.0, reset_origin=False)
   >>> empty.timestamps
   array([])
   >>> empty.values
   array([])




Conversion to IrregularTimeSeries
---------------------------------

:meth:`~RegularTimeSeries.to_irregular` drops gap-fill samples and returns an
:obj:`IrregularTimeSeries` containing only real observations:

.. code-block:: pycon

   >>> irts = signal.to_irregular()
   >>> irts.timestamps
   array([0., 1., 2., 4., 5., 8., 9.])
   >>> irts.values
   array([0.1, 0.4, 0.2, 0.1, 0. , 0.3, 0.5])
   >>> irts.domain.start, irts.domain.end
   array([0., 4., 8.]), array([3., 6., 10.])


The resulting object's domain matches the original gappy series'
multi-interval domain, so the gaps remain explicit even after conversion.
