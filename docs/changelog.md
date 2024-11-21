# Changelog

## Conventions

PiNN follows the [PEP 440] scheme of versioning. Specifically, the
versions are tagged as `major.minor.micro`.  `micro` updates with same
`minor` version are expected to be backward-compactible, i.e., the
models trained on old micro version can be used newer ones.  `minor`
updates can break backward-compatibility, if this happens it should be
documented in this change log.

[PEP 440]: https://peps.python.org/pep-0440/

## v2.x.y

### v2.0.0

- New Network:
    * PiNet2: new equivariant neural network;
- New Models:
    * PiNet-dipole: dipole moment prediction model;
    * PiNet-$\chi$: machine learning charge response kernel model;
- New workflow:
    * nextflow: training pipeline is now managed by [Nextflow](https://www.nextflow.io/docs/latest/index.html)
- New tools:
    * `pinn report`: extract results from work directory or model folder

## v1.x.y

### v1.1.0

- Refactor layers (breaks compatibility with saved models <v1.1.0).

### v1.0.0

- Refactor to TensorFlow 2.

## v0.x.y

### v0.3.0

- Initial public release

