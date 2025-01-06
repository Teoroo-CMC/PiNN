# Quick start
## Installation
**using pip**

```bash
git clone https://github.com/Teoroo-CMC/PiNN.git -b TF2
pip install -e PiNN[gpu]
pinn -h
```

Extra dependencies can be specified:

- `[gpu]` or `[cpu]`: cpu or gpu version of TensorFlow
- `[dev]`: development requirements for testing
- `[doc]`: documentation requirements
- `[extra]`: extra requirements (Jupyter, pymatgen, etc) 


**using container** 

PiNN provides two built docker images, which can be converted to singularity
images without much effort:

```bash
singularity build pinn.sif docker://tecatuu/pinn:master-gpu
./pinn.sif -h
```

- `:master-cpu` is much smaller, it comes without GPU support
- `:master-gpu` is the version with GPU support

Extra dependencies like `Jupyter` are included in the image, for a quick 
development environment:

```bash
singularity run pinn.sif jupyter notebook
```

## Configuration
In PiNN, a model consists of two essential parts, the network and the model. The
network specifies the neural network architecture used to product atomistic
predictions from the input (coordinates and elements information). While the
model interprets the predictions as physical quantities.

In PiNN, the parameters for a model is saved as a `yaml` formatted parameter
file. A minimal example of model parameter file looks like this:

```yaml
model_dir: pinet_potential
model:
  name: potential_model
  params:
    use_force: true
network:
  name: PiNet
  params:
    atom_types: [1, 6, 7, 8, 9]
```

In addition to the network and the model, the training algorithm should be specified
as such:
```yaml
optimizer:
  class_name: EKF
  config:
    learning_rate: 0.03
```

## Using the CLI
PiNN provides a CLI for training and simple dataset operations:

```bash
pinn convert data.traj -o 'trian:8,eval:2'
pinn train -d model -t train.yml -e eval.yml -b 30 --train-steps 100000 params.yml
```

The above command takes a trajectory file `data.traj`, and splits it into two
datasets. Then it takes the parameters from `params.yml`, and trains a model
in the `model` directory.

## Using nextflow

Nextflow enables parallel training of multiple models on a cluster, maximizing the efficient use of computational resources. Follow the steps below to get started:

1. **Install Nextflow**  
   Ensure Nextflow is installed. You can find the installation instructions in the [Nextflow documentation](https://www.nextflow.io/docs/latest/install.html).

2. **Set Up Configuration**  
   Configure your cluster settings in the `nextflow.config` file. Examples are provided in this file to help you get started.

3. **Define Your Data Pipeline**  
   Add your data-loading pipeline in `nextflow/datasets.nf`.

4. **Prepare Your Workflow**  
   Define your workflow in `nextflow/main.nf`. For most tasks, you can reuse the existing processes provided.

5. **Run the Workflow**  
   Execute the workflow with the following command:  
   ```bash
   nextflow run /path/nextflow.config -profile pinet2_qm9_dipole -w /path/work_dir
   ```
   The profile settings in `nextflow.config` are composable. For example, if you want to use a Docker image on the HPC `alvis` to run the `pinet2_qm9_dipole` workflow, you can use the following command:
   ```bash
   nextflow run /path/nextflow.config -profile pinet2_qm9_dipole,alvis,docker -w /path/work_dir
   ```
   This command will automatically pull the Docker image and submit the SLURM tasks.
   Nextflow will now run in the foreground. To avoid interruptions if you close the terminal, it’s recommended to run Nextflow in a `tmux` session or use the `-bg` option to run it in the background. For more details, refer to the [CLI reference](https://www.nextflow.io/docs/latest/reference/cli.html).

## Monitoring
PiNN uses TensorFlow as a backend for training, which means the training log can 
be monitored in real time using Tensorboard:
```bash
tensorboard --log-dir model --port 6006
```

Or with the log inspector of PiNN:
```bash
pinn log model
```


## Using the model
The simplest use case for PiNN models is to use them as ASE calculators. A calculator
can be initialized from the model directory.

```python
import pinn
calc = pinn.get_calc('model')
calc.calculate(atoms)
```
