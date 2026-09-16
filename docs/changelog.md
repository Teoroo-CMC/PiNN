# Changelog

## Conventions

PiNN follows the [PEP 440] scheme of versioning. Specifically, the
versions are tagged as `major.minor.micro`.  `micro` updates with same
`minor` version are expected to be backward-compactible, i.e., the
models trained on old `micro` version can be used newer ones.  `minor`
updates can break backward-compatibility, if this happens it should be
documented in this change log.

[PEP 440]: https://peps.python.org/pep-0440/

## v2.x.y

### v2.1.1

- Training:
    * YAML `settings.dtype` (`float32` / `float64`, default
      **float32**) sets the training float type. The ASE calculator takes
      MACE-style `default_dtype` as a constructor argument only; ``None``
      follows `settings.dtype`.

### v2.1.0

- Backend:
    * TensorFlow window moved from 2.6–2.9 to **2.15** (Python 3.9–3.11, NumPy 1.x).
      This is the last TF release that still ships `tf.estimator` and the legacy
      Keras optimizers (both removed in 2.16), and the first in-window release
      that can use NVIDIA Hopper / GH200 (`sm_90`). See [migration](migration.md).
    * Existing 2.x parameter files and trained models remain usable; YAML
      optimizer names (`Adam`, `SGD`, `EKF`, …) are unchanged.
- Compatibility (TF 2.15):
    * Optimizers now request the *legacy* Keras optimizer automatically so the
      estimator graph-mode `tf.gradients` + `apply_gradients` loop still works.
    * ASE calculator disables `tf.data` prefetch/autotune; otherwise
      `calculate()` could return the previous step's energy/forces.
    * Default ASE raised to **≥3.25.0** in `setup.py` / `environment.yml`
      (Bussi NVT thermostat; 3.22.0 does not provide it). Containers install
      PiNN from those files and do not re-pin ASE.
    * I/O: TFRecord spec from public `dataset.element_spec`; ANI-1 loader uses
      h5py 3 (`dataset[()]`); removed `np.int` / `np.float` aliases.
- Containers:
    * CPU image: `tensorflow/tensorflow:2.15.0` (runtime CLI; no Jupyter).
    * GPU image: NGC `nvcr.io/nvidia/tensorflow:24.03-tf2-py3` (TF 2.15 + CUDA
      12.4, multi-arch amd64/arm64 — the only aarch64 GPU TF 2.15).
    * In-repo `Singularity` / `Singularity.gpu` defs for clusters without Docker.
- CI / packaging:
    * Test matrix is Python 3.9–3.11 × TensorFlow 2.15.
    * Docker CPU image push and docs deploy run only from `Teoroo-CMC/PiNN` on
      `master` / version tags; forks still *build* the CPU image. GPU images
      are built from `Dockerfile.gpu` / `Singularity.gpu` outside GitHub Actions.
    * `setup.py` extras: `.[cpu]` / `.[gpu]` pin `tensorflow>=2.15,<2.16`.
- Workflow:
    * Nextflow `arrhenius` profile (GPU train / CPU data-prep via Apptainer).
    * figshare dataset downloads go through the API (article-zip returned HTTP 202).

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

