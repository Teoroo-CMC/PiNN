# -*- coding: utf-8 -*-
#
__version__ = '1.0.0.dev0'

from pinn.networks import get as get_network
from pinn.models import get as get_model

def get_calc(model_spec, **kwargs):
    """Get a calculator from a trained model.

    The positional argument will be passed to `pinn.get_model`, keyword
    arguments will be passed to the calculator.
    """
    from pinn import get_model
    from pinn.calculator import PiNN_calc
    return  PiNN_calc(get_model(model_spec), **kwargs)
