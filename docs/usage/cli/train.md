#  train

Train a PiNN model given a parameter file.

## Usage

```bash
pinn train params [options]
```

## Options

| Option [shorthand]  | Default       | Description                                               |
|---------------------|---------------|-----------------------------------------------------------|
| `--model-dir [-d]`  | `None`        | model directory (default to 'model' if not set in params) |
| `--train-ds [-t]`   | `'train.yml'` | training set (batched PiNN dataset)                       |
| `--eval-ds [-e]`    | `'eval.yml'`  | evaluation set (batched PiNN dataset)                     |
| `--batch [-b]`      | `None`        | batch size (assume batched dataset by default)            |
| `--(no-)cache`      | `True`        | cache dataset to memory                                   |
| `--(no-)preprocess` | `True`        | preprocess the data                                       |
| `--scatch-dir`      | `None`        | if set, cache the data there instead of RAM               |
| `--train-steps`     | `1e6`         | max training steps                                        |
| `--eval-steps`      | `None`        | evaluation steps (defaults to the whole eval set)         |
| `--shuffle-buffer`  | `100`         | size of shuffle buffer                                    |
| `--log-every`       | `1000`        | log every x steps                                         |
| `--ckpt-every`      | `10000`       | save checkpoint every x steps                             |
| `--max-ckpts`       | `1`           | max number of checkpoints to save                         |
| `--(no-)init`       | `False`       | initialize the params training set                        |
