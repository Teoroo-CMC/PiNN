# Changelog

## Conventions

PiNN follows the [PEP 440] scheme of versioning. Specifically, the
versions are tagged as `major.minor.micro`.  `micro` updates with same
`minor` version are expected to be backward-compactible, i.e., the
models trained on old micro version can be used newer ones.  `minor`
updates can break backward-compatibility, if this happens it should be
documented in this change log.

[PEP 440]: https://peps.python.org/pep-0440/

## v1.x.y

### v1.1.0

- Refactor layers (breaks compatibility with saved models <v1.1.0).

### v1.0.0

- Refactor to TensorFlow 2.

## v0.x.y

### v0.3.0

- Initial public release

