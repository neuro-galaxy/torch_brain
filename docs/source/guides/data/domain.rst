.. currentmodule:: torch_brain.data

Domains
=======

A **domain** is the span of time over which a data object is *valid*. Below, we
create a spike train with 1000 spikes scattered randomly over a 10-second window
and inspect its domain:

.. code:: pycon

   >>> import numpy as np
   >>> from torch_brain.data import IrregularTimeSeries, Interval

   >>> spikes = IrregularTimeSeries(
   ...     timestamps=np.sort(np.random.uniform(0.0, 10.0, size=1000)),
   ...     unit_id=np.random.randint(0, 30, size=1000),
   ... )

   >>> spikes.domain
   Interval(
     start=[1],
     end=[1]
   )
   >>> spikes.domain.start, spikes.domain.end
   (array([0.01541473]), array([9.99930563]))

   >>> min(spikes.timestamps), max(spikes.timestamps)
   (np.float64(0.015414725327748124), np.float64(9.999305627648218))

The domain is itself an :obj:`Interval`. Here it is filled in automatically: by
default, an :obj:`IrregularTimeSeries` sets its domain to the span between its
first and last timestamps.

Often you know more about a recording than its timestamps reveal. For example, if
the probe was switched off between :math:`t=3s` and :math:`t=4s`, no spikes could
have been recorded in that gap. To capture this, we pass the domain directly when
constructing the object:

.. code:: pycon

   >>> spikes = IrregularTimeSeries(
   ...     timestamps=...,
   ...     unit_id=...,
   ...     domain=Interval(start=[0.0, 4.0], end=[3.0, 10.0]),
   ... )

   >>> spikes.domain
   Interval(
     start=[2],
     end=[2]
   )
   >>> spikes.domain.start, spikes.domain.end
   (array([0., 4.]), array([3., 10.]))

The domain is now made up of *two* intervals. A domain can be any set of disjoint
time periods, not just a single one.


Which objects have a domain?
----------------------------

Not every data object has a domain; it depends on whether the object is
inherently temporal.

- :obj:`IrregularTimeSeries` **has a domain.** It defaults to the span of its
  timestamps, but you can also set it explicitly.

- :obj:`RegularTimeSeries` **has a domain.** Its samples are evenly spaced, so
  the domain is computed from the ``sampling_rate``, the number of samples, and
  the ``domain_start`` offset; you never set it by hand. Regular time series can
  also handle missing timepoints; see the :doc:`gappy_regular_ts` guide.

- :obj:`Interval` **does not have a separate domain.** Its ``start`` and ``end``
  times already describe the periods it spans, so there is nothing extra to track.

- :obj:`ArrayDict` **does not have a domain.** It holds non-temporal data, so a
  time span does not apply.

- :obj:`Data` **can have a domain**, if it contains temporal objects. We cover
  this case below.


Domains on ``Data`` objects
---------------------------

A :obj:`Data` object groups several of the objects above into a single recording,
so any :obj:`Data` that holds time-based data must have a domain. We set it
explicitly:

.. code:: pycon

   >>> import numpy as np
   >>> from torch_brain.data import (
   ...     Data, IrregularTimeSeries, RegularTimeSeries, Interval
   ... )

   >>> spikes = IrregularTimeSeries(
   ...     timestamps=np.sort(np.random.uniform(0.0, 10.0, size=1000)),
   ...     unit_id=np.random.randint(0, 30, size=1000),
   ...     domain=Interval(0.0, 10.0),
   ... )
   >>> behavior = RegularTimeSeries(
   ...     sampling_rate=100.0,
   ...     hand_vel=np.random.randn(800, 2),  # 800 samples at 100Hz = 8s long
   ...     domain_start=1.0,  # means the domain of behavior is [1.0, 9.0)
   ... )

   >>> data = Data(
   ...     spikes=spikes,
   ...     behavior=behavior,
   ...     domain=Interval(0.0, 10.0),
   ... )
   >>> data.domain.start, data.domain.end
   (array([0.]), array([10.]))

Passing ``domain="auto"`` tells the constructor to take the union of the domains
of all the objects it contains. Here ``spikes`` spans the full :math:`[0s, 10s)`
while ``behavior`` spans only :math:`[1s, 9s)`, so the union is :math:`[0s, 10s)`:

.. code:: pycon

   >>> data = Data(
   ...     spikes=spikes,      # domain [0.0, 10.0)
   ...     behavior=behavior,  # domain [1.0, 9.0)
   ...     domain="auto",
   ... )
   >>> data.domain.start, data.domain.end
   (array([0.]), array([10.]))


Why domains matter
------------------

When you train a model in **torch_brain**, :ref:`Samplers <samplers_ref>` use
each recording's domain to decide *where* they can draw time-windows from.
Sampling from time periods where the data is absent or corrupted should be
avoided, so removing those periods from the domain is good practice. In
addition, knowing where different recording variables are and aren't valid can
be useful when exploring an existing dataset.
