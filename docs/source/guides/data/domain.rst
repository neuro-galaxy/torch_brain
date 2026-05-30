.. currentmodule:: torch_brain.data

Domains
=======

A **domain** is simply the span of time over which a data object is considered
*valid*. Let's make this concrete. Below we create a spike train with 1000
spikes scattered randomly over a 10-second window, and inspect its domain:

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

The domain is itself an :obj:`Interval`, and here it was filled in automatically:
by default, an :obj:`IrregularTimeSeries` sets its domain to the span between its
first and last timestamps.

This default is just a convenience. Often you know more about the recording than
the timestamps alone reveal. Suppose, for instance, that the probe was switched
off between :math:`t=3` and :math:`t=4`, so no spikes *could* have been recorded
there. You can express that by passing the domain yourself when constructing the
object:

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

Notice that the domain is now made up of *two* intervals. A domain can be any set
of disjoint time periods, not just a single one.


Which objects have a domain?
----------------------------

Not every data object carries a domain; it depends on whether the object is
inherently temporal.

- :obj:`IrregularTimeSeries` **has a domain.** As we saw above, it defaults to the
  span of its timestamps, but you can also set it explicitly.

- :obj:`RegularTimeSeries` **has a domain.** Because its samples are evenly
  spaced, the domain is computed for you from the ``sampling_rate``, the number
  of samples, and the ``domain_start`` offset, so you never set it by hand.
  Although, regular time series *can* handle missing timepoints too; see
  :doc:`gappy_regular_ts`.

- :obj:`Interval` **does not have a separate domain.** Its own ``start`` and ``end``
  times already describe the periods it spans, so there is nothing extra to track.

- :obj:`ArrayDict` **does not have a domain.** It holds non-temporal data, so the
  notion of a time span simply doesn't apply.

- :obj:`Data` **can have a domain**, if it contains temporal objects. We take a closer
  look at this below.


Domains on ``Data`` objects
---------------------------

A :obj:`Data` object groups several of the objects above into a single recording,
so any :obj:`Data` that holds time-based data must be given a domain. You can set
it explicitly:

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

You can also let :obj:`Data` work it out for you. Passing ``domain="auto"``
tells the constructor to take the union of the domains of all the objects it
contains. Here ``spikes`` spans the full :math:`[0, 10)` while ``behavior``
spans only :math:`[1, 9)`, so the union is :math:`[0, 10)`:

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

When you train a model in TorchBrain, :ref:`Samplers <samplers_ref>` look at
each recording's domain to decide *where* they are allowed to draw time-windows
from. You may not want to draw samples from time periods where you know the
data is absent or corrupted. And so, explicitly removing such time periods from
the domain becomes good practice.
