# -*- coding: utf-8 -*-
import tensorflow as tf
from pinn.models.potential import potential_model
from pinn.models.dipole import dipole_model

def get(model_spec, **kwargs):
    import yaml, os
    import numpy as np
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime
    implemented_models = {
        'potential_model': potential_model,
        'dipole_model': dipole_model}

    if isinstance(model_spec, str):
        if tf.io.gfile.exists('{}/params.yml'.format(model_spec)):
            params_file = os.path.join(model_spec, 'params.yml')
        elif tf.io.gfile.exists(model_spec):
            params_file = model_spec
        with FileIO(params_file, 'r') as f:
            model_spec = yaml.load(f, Loader=yaml.Loader)
    # Write the model parameter
    else:
        model_dir = model_spec['model_dir']
        yaml.Dumper.ignore_aliases = lambda *args: True
        to_write = yaml.dump(model_spec)
        params_file = os.path.join(model_dir, 'params.yml')
        if not tf.io.gfile.isdir(model_dir):
            tf.io.gfile.makedirs(model_dir)
        if tf.io.gfile.exists(params_file):
            original = FileIO(params_file, 'r').read()
            if original != to_write:
                tf.io.gfile.rename(params_file, params_file+'.' +
                                   datetime.now().strftime('%y%m%d%H%M'))
        FileIO(params_file, 'w').write(to_write)
    model = implemented_models[model_spec['model']['name']](model_spec, **kwargs)
    return model
