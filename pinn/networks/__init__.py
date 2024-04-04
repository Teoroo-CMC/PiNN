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
    from pinn.networks.pinet2_p5_dot import PiNet2P5Dot
    from pinn.networks.pinet2_p5_prod import PiNet2P5Prod
    from pinn.networks.pinet2_p5_combine import PiNet2P5Combine
    from pinn.networks.pinet2_p5_irrep_dot import PiNet2P5IrrepDot
    from pinn.networks.pinet2_p5_irrep_prod import PiNet2P5IrrepProd
    from pinn.networks.pinet2_p5_irrep_combine import PiNet2P5IrrepCombine
    
    implemented_networks = {
        'PiNet': PiNet,
        'BPNN': BPNN,
        'LJ': LJ,
        'PiNet2': PiNet2,
        'PiNet2P5Dot': PiNet2P5Dot,
        'PiNet2P5Prod': PiNet2P5Prod,
        'PiNet2P5Combine': PiNet2P5Combine,
        'PiNet2P5IrrepDot': PiNet2P5IrrepDot,
        'PiNet2P5IrrepProd': PiNet2P5IrrepProd,
        'PiNet2P5IrrepCombine': PiNet2P5IrrepCombine,
    }
    if isinstance(network_spec, tf.keras.Model):
        return network_spec
    else:
        return  implemented_networks[network_spec['name']](
            **network_spec['params'])
