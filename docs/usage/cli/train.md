#  train

Trian a PiNN model given a parameter file.

## Usage

```bash
pinn trian params [options]
```

## Options

| Option [shorthand] | Default       | Description                                               |
|--------------------|---------------|-----------------------------------------------------------|
| `--model-dir [-d]` | `None`        | Model directory (default to 'model' if not set in params) |
| `--train-ds [-t]`  | `'train.yml'` | Training set (batched PiNN dataset)                       |
| `--eval-ds [-e]`   | `'eval.yml'`  | Evaluation set (batched PiNN dataset)                     |
| `--batch [-b]`     | `None`        | Batch size (assume batched dataset by default)            |
| `--(no)cache`      | `True`        | Cache dataset to memory                                   |
| `--(no)preprocess` | `False`       | Preprocess the data                                       |
| `--scatch-dir`     | `None`        | If set, cache the data there instead of RAM               |
| `--train-steps`    | `1e6`         | Max training steps                                        |
| `--eval-steps`     | `None`        | Evaluation steps (defaults to the whole eval set)         |
| `--shuffle-buffer` | `100`         | Size of shuffle buffer                                    |
| `--max-ckpts`      | `1`           | Max number of checkpoints to save                         |
| `--log-every`      | `1000`        | Log every x steps                                         |
| `--ckpt-every`     | `10000`       | Save checkpoint every x steps                             |
| `--(no)initialzie` | `False`       | Generate atomic dress and fp_range from training set      |
