========
Datasets
========

PiNN uses TensorFlow's dataset class to handle data. Each data point
is a dictionary of tensors. The data point should at least include the
``'coord': coordinates`` and ``'elems': atomic numbers`` as its
features. For training potentials, ``'e_data': energy`` and optionally
``'f_data': forces`` are used as labels.

.. toctree::
   :maxdepth: 1

   datasets/load_data.rst
   datasets/manipulate_data.rst
   datasets/implemented.rst

