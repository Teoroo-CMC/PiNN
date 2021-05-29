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

