# convert

Convert datasets into PiNN formatted TFrecord files.

This command output the dataset files `{output}.yml` and `{output}.tfr`.
Datasets can be split by specifying the ratios in `-o`, e.g. `-o
'train:8,test:2'`. The input dataset will be handled by
[`pinn.io.load_ds`](../datasets.md#api-documentation), with appropriate options.

## Usage

```
pinn convert [options] filename
```

## Options

| Option [shorthand] | Default     | Description                    |
|--------------------|-------------|--------------------------------|
| `--output [-o]`    | `'dataset'` | name of the output dataset     |
| `--format [-f]`    | `'auto'`    | format of input dataset        |
| `--(no-)shuffle`   | `True`      | shuffle dataset when splitting |
| `--seed`           | `0`         | random seed if shuffle is used |
| `--total [-t]`     | `-1`        | total number of samples        |