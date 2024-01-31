# -*- coding: utf-8 -*-

def get(model_spec, **kwargs):
    import yaml, os
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime
    from pinn.models.potential import potential_model
    from pinn.models.dipole import dipole_model
    from pinn.models.atomic_dipole import atomic_dipole_model
    from pinn.models.combined_dipole import combined_dipole_model
    implemented_models = {
        'potential_model': potential_model,
        'dipole_model': dipole_model,
        'atomic_dipole_model': atomic_dipole_model,
        'combined_dipole_model': combined_dipole_model}

    if isinstance(model_spec, str):
        if tf.io.gfile.exists('{}/params.yml'.format(model_spec)):
            params_file = os.path.join(model_spec, 'params.yml')
            with FileIO(params_file, 'r') as f:
                model_spec = dict(yaml.load(f, Loader=yaml.Loader),
                                  model_dir=model_spec)
        elif tf.io.gfile.exists(model_spec):
            params_file = model_spec
            with FileIO(params_file, 'r') as f:
                model_spec = yaml.load(f, Loader=yaml.Loader)
        else:
            raise ValueError(f'{model_spec} does not seem to be a parameter file or model_dir')
    else:
        # we have a dictionary, write the model parameter
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
