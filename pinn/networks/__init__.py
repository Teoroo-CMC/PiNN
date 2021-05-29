# -*- coding: utf-8 -*-

def get(network_spec):
    """Retrieve a PiNN network

    Args:
       network_spec: serialized specification of network, or a Keras model.
    """
    import tensorflow as tf
    from pinn.networks.pinet import PiNet
    from pinn.networks.bpnn import BPNN
    from pinn.networks.lj import LJ
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
