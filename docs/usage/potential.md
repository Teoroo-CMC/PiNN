# Potential Model

## Atomic dress

The absolute value of total energy from ab-initio calculations can often be very
large. To avoid numerical problems, tt is common practice to assign a constant
atomic energy (dress) to each type of atom such that the average energy is
shifted to zero. Such an atomic dress can be generated with
`pinn.utils.get_atomic_dress`

## Loss function 

The loss function in potential model is defined as following:

$$
L = w_e \cdot MSE(e) + w_f \cdot MSE(f) + w_s \cdot MSE (s) + \textrm{regularization terms}
$$

Loss terms are mean squared errors of:

- $e$: energies
- $f$: forces components
- $s$: stress tensor components

It is assumed that the energy, force and stress labels are labelled as
`"e_data"`, `"f_data"`, `"s_data"` in the training set respectively, if they are
to be used in the loss function.

## Parameters

Below a list of additional parameters of the potential model and their
descriptions.

| Parameter           | Default | Description                                                                               |
|---------------------|---------|-------------------------------------------------------------------------------------------|
| `e_dress`           | `{}`    | Atomic Dress                                                                              |
| `e_scale`           | `1`     | The energy scaling during training, this variable defines the energy unit during training |
| `max_energy`        | `False` | When set to a number, exclude energies above this value in loss function                  |
| `use_e_per_atom`    | `False` | Use the per-atom energy in place of total energy in loss function                         |
| `log_e_per_atom`    | `True`  | Log the per-atom energy error, automatically enabled with `use_e_per_atoms=True`          |
| `use_e_weight`      | `False` | Scale the energy loss according to the `'e_weight'` Tensor in the dataset                 |
| `use_force`         | `False` | Include force in loss function                                                            |
| `use_stress`        | `False` | Include stress in loss function                                                           |
| `e_loss_multiplier` | 1       | Weight of energy loss                                                                     |
| `f_loss_multiplier` | 1       | Weight of force loss                                                                      |
| `s_loss_multiplier` | 1       | Weight of stress loss                                                                     |

## Using the potential in ASE calculator

A calculator can be created from a model as simple as:

```Python
from pinn.models import potential_model
from pinn.calculator import PiNN_calc
calc = PiNN_calc(potential_modle('/path/to/model/'))
calc.calculate(atoms)
calc.get_forces()
```

### Units

$\require{mediawiki-texvc}$ 

Following the convention of ASE, the output unit is $eV$ for energy, $eV/\AA$
for forces and $eV/\AA^3$ for stress tensor. Since PiNN does not know about the
unit in the dataset, a `to_eV` parameter is given to convert from the dataset
unit to $eV$.

###  Available results

- `energy`: total energy
- `forces`: forces
- `energies`: atomic contribution to the energy
- `stress`: stress tensor
