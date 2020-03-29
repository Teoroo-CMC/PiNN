# -*- coding: utf-8 -*-
"""``pinn_train`` is a command line utility to train a model.
The program is meant to work with the Google Cloud AI platform,
but it should also run well on local machines.

The trainer effectively runs train_and_evaluate with the given
dataset. To use the trainer, both training data and evaluation data
must be prepared as a tfrecord format. Currently, only training
potentials is supported. To see the available options, run 
``pinn_train --help``

Example usage of ``pinn_train`` on a local machine::

    pinn_trian --model-dir=my_model --params=params.yml \\
               --train-data=train.yml --eval-data=test.yml \\
               --train-steps=1e6 --eval-steps=100 \\
               --cache-data=True

Example usage of ``pinn_train`` on Google Cloud: 

It is recommanded to use our docker image. Since we do not serve it on
Google Container Registery, you'll need to build one yourself (suppose
you have an active Gclound project)::

    gcloud auth configure-docker
    docker pull yqshao/pinn:dev
    docker tag yqshao/pinn:dev gcr.io/my-proj/pinn:cpu
    docker push gcr.io/my-proj/pinn:cpu 

To submit a job on Gcloud::

    gcloud ai-platform jobs submit training my_job_0 \\
           --region europe-west1 --master-image-uri gcr.io/my-proj/pinn:cpu \\
           -- \\
           --model-dir=gs://my-bucket/models/my_model_0 \\
           --params-file=gs://my-bucket/models/params.yml \\
           --train-data=gs://my-bucket/data/train.yml \\
           --eval-data=gs://my-bucket/data/test.yml \\
           --train-steps=1000 --cache-data=True

"""

def trainner(model_dir, params_file,
             train_data, eval_data, train_steps, eval_steps,
             batch_size, preprocess, scratch_dir, cache_data,
             shuffle_buffer, regen_dress):
    import yaml, tempfile, os
    import tensorflow as tf
    from tensorflow.python.lib.io.file_io import FileIO
    from pinn import networks
    from pinn.models import potential_model
    from pinn.utils import get_atomic_dress
    from pinn.io import load_tfrecord, write_tfrecord, sparse_batch

    # Prepare the params or load the model
    with FileIO(params_file, 'r') as f:
        params = yaml.load(f)
    params['model_dir'] = model_dir
    if regen_dress and 'e_dress' in params['model_params']:
        elems = list(params['model_params']['e_dress'].keys())
        dress, _ = get_atomic_dress(load_tfrecord(train_data), elems)
        params['model_params']['e_dress'] = dress

    # Building the datasets
    scratches = []
    def _dataset_fn(fname):
        dataset = load_tfrecord(fname)
        if batch_size is not None:
            dataset = dataset.apply(sparse_batch(batch_size))
        if preprocess:
            if isinstance(params['network'], str):
                network = getattr(networks, params['network'])
            else:
                network = params['network']
            dataset = dataseet.map(network.preprocess)
        if scratch_dir is not None:
            _, tmp = tempfile.mkstemp(dir=scratch_dir)
            scratches.append(tmp)
            write_tfrecord(tmp + '.yml', dataset, pre_fn=pre_fn)
            return lambda: load_tfrecord(tmp + '.yml')
        else:
            return dataset

    train_tmp = lambda: _dataset_fn(train_data)
    eval_fn = lambda: _dataset_fn(eval_data)    
    if scratch_dir is not None and preprocess:
        train_tmp = train_tmp()
        eval_fn = eval_fn()
    if cache_data:
        train_fn = lambda: train_tmp().cache().repeat().shuffle(shuffle_buffer)
    else:
        train_fn = lambda: train_tmp().repeat().shuffle(shuffle_buffer)
        
    # Run
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=train_steps)
    eval_spec  = tf.estimator.EvalSpec(input_fn=eval_fn, steps=eval_steps)
    model = potential_model(params)    
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    
    # Clean up
    for scratch in scratches:
        os.remove(scratch)        
        os.remove(scratch + '.yml')
        os.remove(scratch + '.tfr')

def main():
    import argparse
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
        pass
    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description='Command line tool for training potential model with PiNN.')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='model directory')
    parser.add_argument('--params-file', type=str, required=True,
                        help='path to parameters (.yml file)')
    parser.add_argument('--train-data',  type=str, required=True,
                        help='path to training data (.yml file)')
    parser.add_argument('--eval-data',   type=str, required=True,
                        help='path to evaluation data (.yml file)')
    parser.add_argument('--train-steps', type=int, required=True,
                        help='number of training steps')
    parser.add_argument('--eval-steps',  type=int, 
                        help='number of evaluation steps', default=100)
    parser.add_argument('--batch-size',  type=int,
                        help='Batch size to batch, default to None - data already batched',
                        default=None)
    parser.add_argument('--preprocess',  type=bool,
                        help='Preprocess the data', default=False)
    parser.add_argument('--scratch-dir',  type=str,
                        help='If set in preprocess mode, save the processed dataset to \
                              scratch folder', default=None)
    parser.add_argument('--cache-data',  type=bool,
                        help='cache the training data to memory', default=True)
    parser.add_argument('--shuffle-buffer',  type=int,
                        help='size of shuffle buffer', default=100)    
    parser.add_argument('--regen-dress', type=bool,
                        help='regenerate atomic dress using the training set', default=True)
    
    args = parser.parse_args()
    trainner(args.model_dir, args.params_file,
             args.train_data, args.eval_data,
             args.train_steps, args.eval_steps,
             args.batch_size, args.preprocess,
             args.scratch_dir, args.cache_data, 
             args.shuffle_buffer, args.regen_dress)
