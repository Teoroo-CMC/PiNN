default_adam = {
    'class_name': 'Adam',
    'config': {
        'learning_rate': {
            'class_name': 'ExponentialDecay',
            'config':{
                'initial_learning_rate': 3e-4,
                'decay_steps': 10000,
                'decay_rate': 0.999}},
        'clipnorm': 0.01}}
