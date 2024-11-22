# log

Inspect the status and summary of models.

## Usage

```bash
pinn report models_dir [options]
```

## options
| Option [shorthand] | Default    | Description               |
|--------------------|------------|---------------------------|
| `--keys [-f]`    | ``         | only print results that keywords in model path |
| `--l [--log-name]`         | `'eval.log'` | log file name |
|  `-e [--energy-factor]`   | 1 | energy scale factor, for unit conversion|
|  `-f [--force-factor]`   | 1 | forces scale factor, for unit conversion|
|  `-w [--is_workdir]` | False | try to extract results from work directory |

