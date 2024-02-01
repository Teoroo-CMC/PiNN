# Benchmarks 

## About the benchmark

PiNN collects a small set of benchmarks that are continuously tested
against (see [version convention]). The benchmark datasets and trained
models are accessible from the shared [box folder].

[version convention]: changelog.md/#conventions
[box folder]: https://uppsala.box.com/v/teoroo-cmc-pinn-data

## Latest benchmarks (v1.2.0.dev0)

### QM9[@2014_RamakrishnanDraletal]

Energy MAE: 15.0(std:0.83) meV.

### MD17[@2017_ChmielaTkatchenkoEtAl]

- aspirin:   Energy MAE: 19.47(std:9.31) meV; Force MAE: 13.65(std:1.19) meV/Å.
- ethanol:   Energy MAE: 2.62(std:0.38) meV;  Force MAE: 1.98(std:0.08) meV/Å.
- uracil:    Energy MAE: 7.44(std:4.00) meV;  Force MAE: 6.72(std:0.07) meV/Å.


## Reproducing the benchmark

To manually run the benchmarks with the latest version of PiNN you
will need to have [Nextflow] and [Singularity] installed.

[Nextflow]: https://www.nextflow.io/
[Singularity]: https://docs.sylabs.io/guides/latest/user-guide/

```
nextflow run teoroo-cmc/pinn -r master
```

## For developers

Install PiNN in editable mode and run the benchmark
from the `PiNN` folder.  You will probably need to set up the
development environments on an HPC cluster (e.g. ALVIS):

``` bash
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
python -m venv $HOME/pinn-tf26
source $HOME/pinn-tf26/bin/activate
git clone https://github.com/Teoroo-CMC/PiNN.git && pip install -e PiNN
cd PiNN
```

And run the benchmark using a corresponding profile:

```
export SLURM_ACCOUNT=NAISS2023-5-282
export SALLOC_ACCOUNT=$SLURM_ACCOUNT
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
nextflow run . -profile alvis
```

Adjust the scheduler setup in `nextflow.config` if needed.
