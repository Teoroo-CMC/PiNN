# Dipole Model

The dipole model[@2022_Shao] requires the same dictionary as input as the potential model.
The only difference is the ``model_params`` that can be set. They are listed below
along with their default values.

| Parameter            | Default | Description                                                                      |
|----------------------|---------|----------------------------------------------------------------------------------|
| `max_dipole`         | `False` | When set to a number, exclude dipoles above this value in loss function          |
| `d_scale`            | `1`     | Dipole unit during training                                                      |
| `use_d_per_atom`     | `False` | Use the per-atom dipole in place of total dipole in loss function                |
| `log_d_per_atom`     | `True`  | Log the per-atom dipole error, automatically enabled with `use_d_per_atoms=True` |
| `use_d_weight`       | `False` | Scale the energy loss according to the `'d_weight'` Tensor in the dataset        |
| `use_l2`             | `False` | Include L2 regularization in loss                                                |
| `d_loss_multiplier`  | `1`     | Weight of dipole loss                                                            |
| `l2_loss_multiplier` | `1`     | Weight of l2                                                                     |

## Variants
Since the PiNet2 update several new variants of the dipole model have become available.

- atomic charge (AC) model (`AC_dipole_model`)
- atomic dipole (AD) model (`AD_dipole_model`)
- bond charge model (`BC_R_dipole_model`)
- atomic dipole oxidation state (AD(OS)) model (`AD_OS_dipole_model`)
- AC+AD model (`AC_AD_dipole_model`)
- AC+BC(R) (`AC_BC_R_dipole_model`)
- AD+BC(R) (`AD_BC_R_dipole_model`)

## Usage

Apart from the models containing the atomic dipole (AD), all models are compatible with both PiNet and PiNet2. 

These models can be selected by changing `name` of the model in `model_params` (See overview for further instructions.).

## Parameters
| Parameter            | Default | Description                                                                                                   |
|----------------------|---------|---------------------------------------------------------------------------------------------------------------|
| `max_dipole`         | `False`    | When set to a number, exclude dipoles above this value in loss function                                       |
| `d_scale`            | `1`        | Dipole unit during training                                                                                   |
| `d_unit`             | `1`        | Output unit of dipole during prediction                                                                       |
| `vector_dipole`      | `False`    | Toggle whether to use scalar or vector dipole predictions, should be the same as dipole moment of the dataset |
| `charge_neutrality`  | `True`     | Enable charge neutrality                                                                                      |
| `neutral_unit`       | `'system'` | If charge neutrality is applied, set the charge neutral unit. Choose `'system'` for system neutrality, or  `'water_molecule'` for neutrality per water molecule |               |
| `regularization`     | `True`     | Enable regularization of the interaction prediction, only available for models containing the 'R' term        |
| `use_d_per_atom`     | `False`    | Use the per-atom dipole in place of total dipole in loss function                                             |
| `log_d_per_atom`     | `True`     | Log the per-atom dipole error, automatically enabled with `use_d_per_atoms=True`                              |
| `use_d_weight`       | `False`    | Scale the energy loss according to the `'d_weight'` Tensor in the dataset                                     |
| `use_l2`             | `False`    | Include L2 regularization in loss                                                                             |
| `d_loss_multiplier`  | `1`        | Weight of dipole loss                                                                                         |
| `l2_loss_multiplier` | `1`        | Weight of l2                                                                                                  | 

## Model specifications

### pinn.models.AC.AC_dipole_model
::: pinn.models.AC.AC_dipole_model

### pinn.models.AD.AD_dipole_model
::: pinn.models.AD.AD_dipole_model

### pinn.models.BC_R.BC_R_dipole_model
::: pinn.models.BC_R.BC_R_dipole_model

### pinn.models.AD_OS.AD_OS_dipole_model
::: pinn.models.AD_OS.AD_OS_dipole_model

### pinn.models.AC_AD.AC_AD_dipole_model
::: pinn.models.AC_AD.AC_AD_dipole_model

### pinn.models.AC_BC_R.AC_BC_R_dipole_model
::: pinn.models.AC_BC_R.AC_BC_R_dipole_model

### pinn.models.AD_BC_R.AD_BC_R_dipole_model
::: pinn.models.AD_BC_R.AD_BC_R_dipole_model

