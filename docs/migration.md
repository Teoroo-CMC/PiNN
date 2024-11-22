# Migrating to PiNN 2.x

Since version 2.x, a modularized design, **PiNet2**, has been implemented for equivariant atomistic potential training. **PiNet2** is compatible with **PiNet1**, and you can use the `rank` parameter to specify the desired feature order. To use PiNet, you can either call `pinet` or use `pinet2(rank=2)`—both are functionally equivalent. However, trained models are not interchangeable between the two, meaning you will need to retrain your model if switching versions.

A workflow using [Nextflow](https://www.nextflow.io/docs/latest/index.html) is also integrated, enabling model training on clusters via SLURM or other resource management systems. Examples can be found in the `nextflow.config` file and the [notebook](./notebooks/More_on_training.ipynb).

# Migrating to PiNN 1.x (TF2)

Since version 1.x, PiNN switched to TensorFlow 2 as a backend, this introduces
changes to the API. This document provides information for the changes and
guides for migration.

## New features

**CLI**:
PiNN 1.x introduces a new entry point `pinn` as the command line interface. The
trainer module will be replaced with the `pinn train` sub-command. The CLI also
exposes utilities like dataset conversion for easier usage.

**Parameter file**: 
in PiNN 1.0 the parameter file will serve as a comprehensive input for PiNN
models, the structure of the parameter file is changed, see the documentation
for more information.

**Extended Kalman filter**:
an experimental extended Kalman filter (EKF) optimizer is implemented.


## Notes for developers

- Documentation is now built with mkdocs.
- Documentation is moved to Github pages.
- Continuous integration is moved to Github Actions.
- The Docker Hub repo is now [teoroo/pinn](https://hub.docker.com/repository/docker/teoroo/pinn).

**Datasets**: dataset loaders should be most compatible with PiNN 0.x. With the
TF2 update, dataset may be inspected interactively with eager execution.
Splitting option is simplified (see below), and splitting of `load_tfrecord`
becomes possible.

**Networks**: following the guideline of TF2, networks in PiNN 1.x are new Keras
models and layers becomes Keras layers. This means the PiNN networks can be used
to perform some simple prediction tasks. Note that PiNN models are still
implemented as TensorFlow estimators since they provide a better control over
the training and prediction behavior. Like the design of PiNN 0.x, the models
interpret the predictions of PiNN networks as physical quantities and interface
them to atomic simulation packages.

**Models**:
new helper function `export_mode` and class `MetricsCollector` are implemented to
simplify the implementation of models, see the source of [dipole
model](https://github.com/Teoroo-CMC/PiNN/blob/master/pinn/models/dipole.py) for an
example.

## Breaking changes
- Models trained in PiNN 0.x will not be usable in PiNN 1.x.
- Model parameters need to be adapted to the new parameter format.
- For dataset loaders `load_*`:
    + the `split` argument is renamed to `splits`;
    + splitting is disabled by default;
    + nested splits like `{'train':1, 'test':[1,2,3]}` is not supported anymore.
- `format_dict` is renamed as `ds_spec` to be consistent with
  [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/data/DatasetSpec).
