Overview
========

Data is represented as `tf.Dataset` objects in PiNN. PiNN provides the reader
and writers for a range of file formats used for training atomic neural
networks. In addition, PiNN provide functions to download and load some commonly
benchmarked datasets.

Below is a list of the available file formats and datasets in PiNN:

File formats
------------

Below is a list of file writer/readers implemented in PiNN

+----------+------------------------------------------------------------------------------------+--------------+
| Format   | Descritpion                                                                        | Capabilities |
+==========+====================================================================================+==============+
| TFRecord | TFRecord format                                                                    | RW           |
+----------+------------------------------------------------------------------------------------+--------------+
| RuNNer   | Dataset format used in the RuNNer http://www.uni-goettingen.de/de/560580.html code | R            |
+----------+------------------------------------------------------------------------------------+--------------+
| Numpy    | Numpy array datasets                                                               | R            |
+----------+------------------------------------------------------------------------------------+--------------+
| XYZ      | xyz format                                                                         | R            |
+----------+------------------------------------------------------------------------------------+--------------+
| CP2K     | CP2K output files                                                                  | R            |
+----------+------------------------------------------------------------------------------------+--------------+


Public datasets
---------------

+---------------+-------------+-----------+
| Dataset       | Description | Reference |
+===============+=============+===========+
| QM9 dataset   |             |           |
+---------------+-------------+-----------+
| ANI-1 dataset |             |           |
+---------------+-------------+-----------+
