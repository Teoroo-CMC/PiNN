# -*- coding: utf-8 -*-
#
__version__ = '1.0.1.dev'

from pinn.networks import get as get_network
from pinn.models import get as get_model

def get_calc(model_spec, **kwargs):
    """Get a calculator from a trained model.

    The positional argument will be passed to `pinn.get_model`, keyword
    arguments will be passed to the calculator.
    """
    import tensorflow as tf
    from pinn import get_model
    from pinn.calculator import PiNN_calc
    if isinstance(model_spec, tf.estimator.Estimator):
        model = model_spec
    else:
        model = get_model(model_spec)
    return  PiNN_calc(model, **kwargs)
