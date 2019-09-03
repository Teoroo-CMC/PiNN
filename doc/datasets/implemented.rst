===================
Implemented loaders
===================

For common QML datasets, we provide the function to directly load the
dataset. The loaders might be limited by IO, but if you have enough
memory, you can simply cache the dataset with ``dataset.cache()`` or
convert them to tfrecords.

The RuNNer Format
-----------------

RuNNer data (used by the RuNNer code:
http://www.uni-goettingen.de/en/560580.html) has the format::

    begin
    lattice float float float
    lattice float float float
    lattice float float float
    atom floatcoordx floatcoordy floatcoordz int_atom_symbol floatq 0  floatforcex floatforcey floatforcez
    atom 1           2           3           4               5      6  7           8           9
    energy float
    charge float
    comment arbitrary string
    end

The order of the lines within the begin/end block are arbitrary The
coordinates are given in bohr, the charges in atomic units, the energy
in Ha, and the force components in Ha/bohr

.. autofunction:: pinn.io.load_runner

QM9 dataset
-----------

The QM9 dataset includes many computed properties for 134 stable
organic molecules.  See
ref. :cite:`ramakrishnan_dral_dral_rupp_anatole_von_lilienfeld_2017`
for more details

The default behavior here is to label the internal energy "U0" as
"e_data".  This behavior can be tweaked with the :code:`label_map`
parameter.

.. autofunction:: pinn.io.load_qm9

ANI-1 dataset
-------------

The ANI-1 dataset consists of 20M off-equilibrium DFT energies for
orginic molecules.  See ref. :cite:`smith_isayev_roitberg_2017` for
more details

.. autofunction:: pinn.io.load_ani


Numpy dataset
-------------

Another easy way to generate you own dataset is to store the data as a
dictionary of numpy arrays. See how it's done in the :doc:`toy problem
<../notebooks/4_Toy_LJ_with_three_atoms>`.
