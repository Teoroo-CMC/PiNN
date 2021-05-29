# convert

Convert datasets into PiNN formated TFrecord.

## Usage

```
$ pinn convert [options] filename
```

## Options

| Option [shorthand] | Default     | Description                      |
|--------------------|-------------|----------------------------------|
| `--output [-o]`    | `'dataset'` | name of the output dataset       |
| `--format [-f]`    | `'auto'`    | format of input dataset          |
| `--(no-)shuffle`   | `False`     | shuffle dataset before splitting |
| `--batch`          | `False`     | convert dataset into minibatches |
| `--seed`           | `0`         | random seed if shufffle is used  |

## Supported formats

- files ending with `.yml` are assumed to the PiNN tfrecord format (`-f pinn`).
- other files will be parsed to `ase.io.load`, energy and force data will be
  labelled as `e_data` and `f_data` respectively if exist.
- the PBC and dataset must be consistent across the dataset.

