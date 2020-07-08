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

def get_network(network, **kwargs):
    """Retrieve a PiNN network

    Args:
        network: string, network, or a Keras model.
        **kwargs: keyword arguments for the network.
            (ignored when network is a Keras model)
    """
    implemented_networks = {
        'PiNet': PiNet,
        'BPNN': BPNN,
        'LJ': LJ
    }
    if network in implemented_networks:
        return implemented_networks[network](**kwargs)
    elif isinstance(network, type):
        return network(**kwargs)
    elif isinstance(network, tf.keras.Model):
        return network
