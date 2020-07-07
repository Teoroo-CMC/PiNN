# Use PiNN with Ray Tune

>  [Tune](https://ray.readthedocs.io/en/latest/tune.html) is a library for
>  hyperparameter tuning at any scale.

We have some helpers to tune the hyperparameters with the Tune library.

## A working example

Suppose we would like to tune the hyperparameters for say, learning
rate, number of Pi blocks and the cutoff radius for a PiNet potential.
We need to first declare the space we would like to search, with Tune.

```python
from ray import tune
config = {
'lr': tune.grid_search([3e-4, 1e-4, 3e-5]),
'rc': tune.grid_search([3.5, 4.5, 5.0, 5.5]),
'depth': tune.grid_search([4, 5, 6])}
``` 

Then we need to define a `train_fn`. Given a config it should return several
objects, a model (as an estimator), the train_spec and eval_spec, and a reporter
function which reports the metrics to Tune.

The reporter should take the evaluation output as input, and return a dictionary
of metrics. The naming does not matter, but note that Tune expects a
`reward_attr` which increases during training to discriminate models, so you
should define things like `accuracy` instead of `error`.

```python
def train_fn(c, ckpt=None):
    import yaml, os
    import tensorflow as tf
    from pinn.io import load_tfrecord
    from pinn.models import potential_model
    # parameter builder
    params = {
    # Tune will take charge of the running path,
    # so model_dir can be a relative directory
    'model_dir': './model_dir/',
        'network': 'pinet',
        'network_params': {
        'depth': c['depth'],
            'atom_types': [1,6,7,8,9]},
        'model_params': {
            'learning_rate': c['lr'],
            'en_scale': 1}}
    # disable checkpoint cleaning, and specify report frequency
    # note we report to Tune whenever a checkpoint is saved
    config = tf.estimator.RunConfig(
    save_checkpoints_secs=300,
        keep_checkpoint_max=None, 
        keep_checkpoint_every_n_hours=None)
    # Returning things
    model = potential_model(params, config=config, warm_start_from=ckpt)
    train_fn = lambda: load_tfrecord('/path/to/train.yml')
    test_fn = lambda: load_tfrecord('/path/to/test.yml')	
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=1e6)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_fn)
    reporter = lambda eval_out: {'accuracy': 1./eval_out['METRICS/E_MAE']}
    return model, train_spec, eval_spec, reporter
``` 

Now you can use the TuneTrainable function to feed your `train_fn`
to Tune as a trainable.

```Python
import ray
from pinn.utils import TuneTrainable	  
# Define the trainable from train_fn
trainable = TuneTrainable(train_fn)
# Start ray cluster, use all the GPUs on this machine
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
ray.shutdown()
ray.init()
# Searching strategy of Ray Tune	  
search_alg = tune.suggest.BasicVariantGenerator(shuffle=True)
scheduler = tune.schedulers.HyperBandScheduler(
    time_attr='time_total_s', reward_attr='accuracy', max_t=3600)
# Run jobs with Ray tune
tune.run(trainable, name='test_tunning', config=config,
             resources_per_trial={'cpu':5, 'gpu':1},
             search_alg=search_alg, scheduler=scheduler,
             local_dir="/tmp/test_tuning",
             num_samples=2, verbose=1)
```


We have a [notebook example](../notebooks/Tune_visualize.ipynb) for visualizing tuning
results from Tune.
   

## How this works

The `TuneTrainable` function is actually a general wrapper for estimators to
work with Ray Tune. It runs the train and evaluate but stops the estimator
whenever a checkpoint is saved and reports the metrics to Tune.

This is a bit more elaborate than adding a reporter hook to the estimator, and
slightly slower since the training must restart from checkpoints. But the gain
is that Tune can now stop and restore trainings when it wants, which is required
by of some of Tune's
[schedulers](https://ray.readthedocs.io/en/latest/tune-schedulers.html).
