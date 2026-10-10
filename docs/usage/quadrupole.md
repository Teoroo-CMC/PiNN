# Quadrupole Model

The quadrupole model[@Muuga_2026] works similarly to the dipole model.
The ``model_params`` can be set, and are listed below along with their default values.

| Parameter            | Default | Description                                                                      |
|----------------------|---------|----------------------------------------------------------------------------------|
| `max_quadrupole`        | `False` | When set to a number, exclude quadrupoles above this value in loss function          |
| `quad_scale`            |   `1`   | Quadrupole unit during training                                                      |
| `use_quad_per_atom`     | `False` | Use the per-atom quadrupole in place of total quadrupole in loss function                |
| `log_quad_per_atom`     | `True`  | Log the per-atom quadrupole error, automatically enabled with `use_d_per_atoms=True` |
| `use_quad_weight`       | `False` | Scale the energy loss according to the `'d_weight'` Tensor in the dataset        |
| `use_l2`                | `False` | Include L2 regularization in loss                                                |
| `quad_loss_multiplier`  |   `1`   | Weight of quadrupole loss                                                            |
| `l2_loss_multiplier`    |   `1`   | Weight of l2                                                                     |

## Variants
There are different variants of quadrupole and quadrupole-quadrupole models available.

- atomic charge (AC) model (`AC_quadrupole_model`)
- atomic charge dipole-quadrupole (AC-DQ) model (`AC_dipole-quadrupole_model`)
- AC+AD dipole-quadrupole model (`AC_AD_dipole_quadrupole_model`)

## Usage

Apart from the models containing the atomic dipole (AD), all models are compatible with both PiNet and PiNet2. 

These models can be selected by changing `name` of the model in `model_params` (See overview for further instructions.).

## Parameters
| Parameter            | Default | Description                                                                                                   |
|----------------------|---------|---------------------------------------------------------------------------------------------------------------|
| `max_quadrupole`       | `False` | When set to a number, exclude quadrupoles above this value in loss function                                       |
| `quad_scale`           |   `1`   | Quadrupole unit during training                                                                                   |
| `quad_unit`            |   `1`   | Output unit of quadrupole during prediction                                                                       |
| `vector_quadrupole`    | `True` | Toggle whether to use scalar or vector quadrupole predictions, should be the same as quadrupole moment of the dataset |
| `charge_neutrality`    | `True`  | Enable charge neutrality                                                                                      |
| `neutral_unit`         |`'system'`| If charge neutrality is applied, set the charge neutral unit. Choose `'system'` for system neutrality, or  `'water_molecule'` for neutrality per water molecule |               |
| `regularization`       |  `True`   Enable regularization of the interaction prediction, only available for models containing the 'R' term        |
| `use_quad_per_atom`    | `False` | Use the per-atom quadrupole in place of total quadrupole in loss function                                             |
| `log_quad_per_atom`    | `True`  | Log the per-atom quadrupole error, automatically enabled with `use_quad_per_atoms=True`                              |
| `use_quad_weight`      | `False` | Scale the energy loss according to the `'d_weight'` Tensor in the dataset                                     |
| `use_l2`               | `False` | Include L2 regularization in loss                                                                             |
| `quad_loss_multiplier` |   `1`   | Weight of quadrupole loss                                                                                         |
| `l2_loss_multiplier`   |   `1`   | Weight of l2                                                                                                  | 

## Model specifications

### pinn.models.AC_quadrupole_model
::: pinn.models.AC_quadrupole_model

### pinn.models.AC_dipole_quadrupole.py
::: pinn.models.AC_dipole_quadrupole.py

### pinn.models.AC_AD_dipole_quadrupole.py
::: pinn.models.AC_AD_dipole_quadrupole.py


