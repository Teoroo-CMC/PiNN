# -*- coding: utf-8 -*-
#
__version__ = '2.0.0'

from pinn.networks import get as get_network
from pinn.models import get as get_model
from pinn import report

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

def get_available_networks():
    print("Available networks:")
    print("  - PiNet")
    print("  - PiNet2")
    print("  - BPNN")
    print("  - LJ")

def get_available_models():
    print("Available models:")
    print("  - potential_model")
    print("  - dipole_model")
    print("  - AC_dipole_model")
    print("  - AD_dipole_model")
    print("  - BC_R_dipole_model")
    print("  - AD_OS_dipole_model")
    print("  - AC_AD_dipole_model")
    print("  - AC_BC_R_dipole_model")
    print("  - AD_BC_R_dipole_model")
