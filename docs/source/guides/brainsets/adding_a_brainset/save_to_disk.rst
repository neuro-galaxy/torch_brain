Save Sessions to Disk
=====================

Write each :class:`~torch_brain.data.Data` object to an H5 file under
``self.processed_dir`` inside :meth:`~torch_brain.pipeline.BrainsetPipeline.process`.

.. code-block:: python

   import h5py
   from torch_brain.data import Data, serialize_fn_map

   def process(self, fpath):
       self.update_status("Loading file")
       ...

       output_file_path = self.processed_dir / f"{session_id}.h5"
       if output_file_path.exists() and not self.args.reprocess:
           return

       self.update_status("Extracting neural activity")
       ...
       data = Data(...)  # see :doc:`build_session`

       self.update_status("Storing")
       with h5py.File(output_file_path, "w") as file:
           data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

We can safely skip writing when the output file already exists unless ``--reprocess`` is set.