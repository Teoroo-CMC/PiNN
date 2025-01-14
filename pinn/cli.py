#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
import operator
from pathlib import Path
import random
import subprocess
import click, pinn, os
from pinn.report import report_log
from typing import List
import numpy as np
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CONTEXT_SETTINGS = dict()

@click.group()
def main():
    """PiNN CLI - Command line interface for PiNN"""
    pass

@click.command()
def version():
    click.echo(f'PiNN version: {pinn.__version__}')

@click.command(name='convert', context_settings=CONTEXT_SETTINGS,
               options_metavar='[options]', short_help='convert datasets')
@click.argument('filename', metavar='filename', nargs=1)
@click.option('-f', '--fmt', metavar='', default='auto', show_default=True)
@click.option('-o', '--output', metavar='', default='dataset', show_default=True)
@click.option('--shuffle/--no-shuffle', metavar='', default=True, show_default=True)
@click.option('--seed', metavar='', default='0',  type=int, show_default=True)
@click.option('-t', '--total', metavar='', default='-1',  type=int, show_default=True)
def convert(filename, fmt, output, shuffle, seed, total):
    """Convert or split dataset to PiNN formatted tfrecord files

    See the documentation for more detailed descriptions of the options
    https://Teoroo-CMC.github.io/PiNN/latest/usage/cli/convert/
    """
    from pinn.io import load_ds, write_tfrecord
    assert total >= -1, \
        ValueError('Total used data must greater than 0, -1 to retrieve all data')
    if ':' not in output: # single output
        ds = load_ds(filename, fmt=fmt)
        write_tfrecord(f'output.yml', ds.take(total))
    else:
        splits = {
            s.split(':')[0]: float(s.split(':')[1])
            for s in output.split(',')}
        ds = load_ds(filename, fmt=fmt, splits=splits, shuffle=shuffle, seed=seed)
        if total == -1:  # If count is -1, or if count is greater than the size of this dataset, return whole dataset
            for k, v in ds.items():
                write_tfrecord(f'{k}.yml', v)
        else:  # or scale down each dataset
            total_splits = reduce(operator.add, splits.values())
            for k, v in ds.items():
                take_n = int(total * splits[k] / total_splits)
                write_tfrecord(f'{k}.yml', v.take(take_n))

@click.command(name='train', context_settings=CONTEXT_SETTINGS,
               options_metavar='[options]', short_help='train model')
@click.argument('params', metavar='params', nargs=1)
@click.option('-d', '--model-dir', metavar='', default=None, help="[default: None (get from params)]")
@click.option('-t', '--train-ds', metavar='', default='train.yml', show_default=True)
@click.option('-e', '--eval-ds', metavar='', default='eval.yml', show_default=True)
@click.option('-b', '--batch', metavar='', type=int, default=None, help="[default: None (keep as input)]")
@click.option('--cache/--no-cache', metavar='', default=True, show_default=True)
@click.option('--preprocess/--no-preprocess', metavar='', default=True, show_default=True)
@click.option('--scratch-dir', metavar='', default=None, help='[default: None (cache in RAM)]')
@click.option('--train-steps', metavar='', default=1000000, type=int, show_default=True)
@click.option('--eval-steps', metavar='', default=None, type=int, help='[default: None (entire eval set)]')
@click.option('--shuffle-buffer', metavar='', default=100, type=int, show_default=True)
@click.option('--log-every', metavar='', default=1000, type=int, show_default=True)
@click.option('--ckpt-every', metavar='', default=10000, type=int, show_default=True)
@click.option('--max-ckpts', metavar='', default=1, type=int, show_default=True)
@click.option('--early-stop', metavar='', type=str, default=None, help="[default: None]")
@click.option('--init/--no-init', metavar='', default=False, show_default=True)
@click.option('--eval-bs', metavar='', type=int, default=None, help="default same with train batch size")
def train(params, model_dir, train_ds, eval_ds, batch, cache, preprocess,
          scratch_dir, train_steps, eval_steps, shuffle_buffer,
          max_ckpts, log_every, ckpt_every, early_stop, init, eval_bs):
    """Train a model with PiNN.

    See the documentation for more detailed descriptions of the options
    https://Teoroo-CMC.github.io/PiNN/latest/usage/cli/train/
    """
    import yaml, warnings
    import tensorflow as tf
    from shutil import rmtree
    from tempfile import mkdtemp, mkstemp
    from tensorflow.python.lib.io.file_io import FileIO
    from pinn import get_model, get_network
    from pinn.utils import init_params
    from pinn.io import load_tfrecord, sparse_batch
    index_warning = 'Converting sparse IndexedSlices'
    warnings.filterwarnings('ignore', index_warning)
    tf.get_logger().setLevel('ERROR')

    with FileIO(params, 'r') as f:
        params = yaml.load(f, Loader=yaml.Loader)
    if model_dir is not None:
        params['model_dir'] = model_dir

    if init:
        ds = load_tfrecord(train_ds)
        init_params(params, ds)
    if eval_bs is None:
        eval_bs = batch

    if scratch_dir is not None:
        scratch_dir = mkdtemp(prefix='pinn', dir=scratch_dir)
    def _dataset_fn(fname, batch):
        dataset = load_tfrecord(fname)
        if batch is not None:
            dataset = dataset.apply(sparse_batch(batch))
        if preprocess:
            def pre_fn(tensors):
                with tf.name_scope("PRE") as scope:
                    network = get_network(params['network'])
                    tensors = network.preprocess(tensors)
                return tensors
            dataset = dataset.map(pre_fn)
        if cache:
            if scratch_dir is not None:
                cache_dir = mkstemp(dir=scratch_dir)
            else:
                cache_dir = ''
            dataset = dataset.cache(cache_dir)
        return dataset

    train_fn = lambda: _dataset_fn(train_ds, batch).repeat().shuffle(shuffle_buffer)
    eval_fn = lambda: _dataset_fn(eval_ds, eval_bs)
    config = tf.estimator.RunConfig(keep_checkpoint_max=max_ckpts,
                                    log_step_count_steps=log_every,
                                    save_summary_steps=log_every,
                                    save_checkpoints_steps=ckpt_every)

    model = get_model(params, config=config)
    if early_stop:
        stops = {s.split(':')[0]: float(s.split(':')[1])
                 for s in early_stop.split(',')}
        hooks = [tf.estimator.experimental.stop_if_no_decrease_hook(
            model, k, v) for k,v in stops.items()]
    else:
        hooks=None
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=train_steps, hooks=hooks)
    eval_spec  = tf.estimator.EvalSpec(input_fn=eval_fn, steps=eval_steps)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    if scratch_dir is not None:
        rmtree(scratch_dir)

@click.command(name='log', context_settings=CONTEXT_SETTINGS,
               options_metavar='[options]', short_help='inspect training logs')
@click.argument('logdir', metavar='logdir', nargs=1)
@click.option('-t','--tag', metavar='', default='RMSE', show_default=True)
@click.option('-f','--fmt', metavar='', default='%14.6e ', show_default=True)
def log(logdir, tag, fmt):
    import numpy as np
    from glob import glob
    from sys import stdout
    from warnings import warn
    from itertools import chain
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    files = sorted(glob(f'{logdir}/events.out.*'), key=os.path.getmtime)
    logs = {}
    events = chain(*[LegacyEventFileLoader(log).Load() for log in files])
    for event in events:
        for v in event.summary.value:
            if tag not in v.tag:
                continue
            if v.tag not in logs.keys():
                logs[v.tag] = []
            logs[v.tag].append([event.step, v.simple_value])
    logs = {k: np.array(v) for k,v in logs.items()}
    keys = sorted(list(logs.keys()))
    steps = [logs[k][:,0] for k in keys]
    data = [logs[k][:,1] for k in keys]
    steps, rows = np.unique(np.concatenate(steps), return_inverse=True)
    cols = np.concatenate([np.full_like(v, i, np.int) for i, v in enumerate(data)])

    tmp = np.full([len(steps), len(logs.values())], np.nan)
    tmp[rows, cols] = np.concatenate(data)
    out = np.concatenate([steps[:,None], tmp], axis=1)
    header = ('Step',*[k.replace('METRICS/','') for k in keys])
    np.savetxt(stdout, out, '%-9d '+fmt*tmp.shape[1],
               header='  '.join(header))

@click.command(name='report', context_settings=CONTEXT_SETTINGS,
                options_metavar='[options]', short_help='generate report')
@click.argument('publish_dir', metavar='publish_dir', nargs=1)
@click.option('-k', '--keys', metavar='', default='', multiple=True)
@click.option('-l', '--log-name', metavar='', default='eval.log', show_default=True)
@click.option('-e', '--energy-factor', metavar='', default=1, help='energy convert factor', type=float)
@click.option('-f', '--force-factor', metavar='', default=0, help='energy convert factor', type=float)
@click.option('-w', '--is_workdir', is_flag=True, default=False, help='extract result directly from nextflow work directory')
def report(publish_dir:str, keys:List[str], log_name:str, energy_factor:str, force_factor:str, is_workdir: bool):
    publish_dir = Path(publish_dir)
    log_paths = publish_dir.glob(f'**/{log_name}')
    result = {}  # path: [e_mae, f_mae]
    fields = []
    for log_path in log_paths:
        if is_workdir:
            dirlist = [d for d in log_path.parent.iterdir() if d.is_dir()]
            if len(dirlist) > 1:
                raise ValueError('can not make a guess', dirlist)
            model_name = str(dirlist[0])
        else:
            model_name = str(log_path.parent.stem)
        if not all(map(lambda key: key in model_name, keys)):
            continue
        with open(log_path) as f:
            header = f.readline()
        fields = header.split()[1:]  # skip #
        log = np.loadtxt(log_path)
        e_mae = log[-1, 2]
        result[model_name] = {field: log[-1, i] for i, field in enumerate(fields)}
    reduced_data = defaultdict(dict)
    for key, value in result.items():
        prefix, _, _id = key.split('/')[-1].rpartition('-')
        for k, v in value.items():
            if k not in reduced_data[prefix]:
                reduced_data[prefix][k] = []
            reduced_data[prefix][k].append(v)
    if not fields:
        raise FileNotFoundError("None matched record found")

    for prefix, data in reduced_data.items():
        msg = {'model': prefix.split('/')[-1]}
        for key, value in data.items():
            if key == 'Step':
                msg['Step'] = str(int(value[0]))
            elif key == 'checkpoint_path':
                continue
            else:
                vv = np.array(value)
                if key.startswith("E"):
                    vv *= energy_factor
                elif key.startswith("F"):
                    vv *= force_factor
                msg[f'{key}_MEAN'] = f"{float(vv.mean()):.4f}"
                msg[f'{key}_STD'] = f"{float(vv.std()):.4f}"

        print(msg)

main.add_command(convert)
main.add_command(train)
main.add_command(log)
main.add_command(version)
main.add_command(report)

if __name__ == '__main__':
    main()
