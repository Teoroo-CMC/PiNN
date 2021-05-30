# convert

Convert datasets into PiNN formated TFrecord.

This commands output the dataset files `{output}.yml` and `{output.tfr}`. Datset
can be splitted by specifying the ratios in `-o`, e.g. `-o 'train:8,test:2'`.
The input dataset will be handled by
[`pinn.io.load_ds`](../datasets.md#api-documentation), with appropriate options.

## Usage

```
pinn convert [options] filename
```

## Options

| Option [shorthand] | Default     | Description                      |
|--------------------|-------------|----------------------------------|
| `--output [-o]`    | `'dataset'` | name of the output dataset       |
| `--format [-f]`    | `'auto'`    | format of input dataset          |
| `--(no-)shuffle`   | `True`      | shuffle dataset when splitting |
| `--seed`           | `0`         | random seed if shufffle is used  |

