# log

Inspect the training log of a model, prints training metrics in plain text.

## Usage

```bash
pinn log [options] logdir
```

## options
| Option [shorthand] | Default    | Description               |
|--------------------|------------|---------------------------|
| `--tag [-t]`       | `'RMSE'`   | tags to print             |
| `--fmt [-f]`       | `'%14.6e'` | format string for metrics |
