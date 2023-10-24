# Benchmarks 

## About the benchmark

PiNN collects a small set of benchmarks that are continuously tested
against (see [version convention]). The benchmark datasets and trained
models are accessible from the shared [box folder].

[version convention]: changelog.md/#conventions
[box folder]: https://uppsala.box.com/v/teoroo-cmc-pinn-data

## Latest benchmarks (v1.2.0)

TO-BE-GENERATED

## Run manually

To manually run the benchmarks with the latest version of PiNN you
will need to have [Nextflow] and [Singularity] installed.

[Nextflow]: https://www.nextflow.io/
[Singularity]: https://docs.sylabs.io/guides/latest/user-guide/

```
nextflow run teoroo-cmc/pinn -r master
```

For developers, install PiNN in editable mode and run the benchmark
from the `PiNN` folder:

```
# git clone https://github.com/Teoroo-CMC/PiNN.git && pip install -e PiNN
nextflow run .
```
