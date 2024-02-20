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
    from pinn.networks.pinet2 import PiNet2
    from pinn.networks.pinet2_energy import PiNet2_energy
    implemented_networks = {
        'PiNet': PiNet,
        'BPNN': BPNN,
        'LJ': LJ,
        'PiNet2': PiNet2,
        'PiNet2_energy': PiNet2_energy
    }
    if isinstance(network_spec, tf.keras.Model):
        return network_spec
    else:
        return  implemented_networks[network_spec['name']](
            **network_spec['params'])
