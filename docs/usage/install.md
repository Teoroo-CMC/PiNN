# Installation

## With pip

```
git clone https://github.com/yqshao/PiNN.git -b TF2
pip install -e PiNN
pinn -h
```

Extra dependencies are avaiable:
- `[dev]`: development requirements (testing suit)
- `[doc]`: developemnt requirements (documentation builder)
- `[extra]`: extra requirements (Jupyter, pymatgen, etc) 

## With container

PiNN provides three built docker images, which can be converted to 
singularity images without much effort:

```python
singularity build pinn.sif docker://yqshao/pinn:tf2
./pinn.sif -h
```

Three images are provided:
- `:tf2` is the most compact, it comes without GPU support
- `:tf2-gpu` is the version with GPU support
- `:tf2-full` contains all extra requirements 

