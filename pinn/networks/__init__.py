# -*- coding: utf-8 -*-
"""Networks defines the structure of a model

Networks should be pure functions except during pre-processing, while
a nested tensors gets updated as input. 
Networks should not define the goals/loss of the model, 
to allows for the usage of same network structure for different tasks.
"""
from pinn.networks.pinn import pinn_network
from pinn.networks.bpnn import bpnn_network
from pinn.networks.lj import lj
