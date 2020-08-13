# -*- coding: utf-8 -*-
"""Networks defines the structure of a model

Networks should be pure functions except during pre-processing, while
a nested tensors gets updated as input. 
Networks should not define the goals/loss of the model, 
to allows for the usage of same network structure for different tasks.
"""
import tensorflow as tf
from pinn.networks.pinet import PiNet
from pinn.networks.bpnn import BPNN
from pinn.networks.lj import LJ

def get(network_spec):
    """Retrieve a PiNN network

    Args:
       network_spec: serialized specification of network, or a Keras model.
    """
    implemented_networks = {
        'PiNet': PiNet,
        'BPNN': BPNN,
        'LJ': LJ
    }
    if isinstance(network_spec, tf.keras.Model):
        return network_spec
    else:
        return  implemented_networks[network_spec['name']](
            **network_spec['params'])
