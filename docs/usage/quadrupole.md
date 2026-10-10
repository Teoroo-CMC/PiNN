# Quadrupole Model

The quadrupole model[@Muuga_2026] works similarly to the dipole model. The traceless quadrupole moments of the format [$Q_{xx}$, $Q_{yy}$, $Q_{zz}$, $Q_{xy}$, $Q_{xz}$, $Q_{yz}$] are predicted using atomic charges as latent variables.


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

## Dipole-quadrupole moments

The dipole-quadrupole model fits the dipole and quadrupole of the inputs to the atomic charges
and predicts dipoles and quadrupoles. The AC+AD dipole-quadrupole model uses a contribution from atomic dipoles for dipole prediction.

The dipole moment is expressed as

\\[ \boldsymbol{\mu}(\mathbf{r}) = \sum_i q_i \mathbf{r}\_i \\]

The quadrupole tensor is defined as

\\[ Q'\_{\alpha\beta}(\mathbf{r}) = \sum_i q_i \left( 3r\_{\alpha i}r\_{\beta i} - \delta\_{\alpha\beta}|\mathbf{r}\_i|^2 \right) \\]

where:

- \\(q_i\\) is the atomic charge of atom \\(i\\).
- \\(\mathbf{r}\_i\\) is the position vector of atom \\(i\\).
- \\(r\_{\alpha i}\\) and \\(r\_{\beta i}\\) are the Cartesian components of \\(\mathbf{r}\_i\\).
- \\(\alpha,\beta \in \\{x,y,z\\}\\).
- \\(\delta\_{\alpha\beta}\\) is the Kronecker delta.

The traceless quadrupole tensor is defined in terms of the quadrupole tensor \\(Q\_{\alpha\beta}\\) as

\\[ \boxed{ Q'\_{\alpha\beta} = 3Q\_{\alpha\beta} - \operatorname{tr}(Q)\delta\_{\alpha\beta} } \\]

where \\(\operatorname{tr}(Q)\\) denotes the trace of the quadrupole tensor:

\\[ \operatorname{tr}(Q) = \sum\_{\gamma \in \\{x,y,z\\}} Q\_{\gamma\gamma} \\]

## Usage

Apart from the models containing the atomic dipole (AD), all models are compatible with both PiNet and PiNet2. 

These models can be selected by changing `name` of the model in `model_params` (See overview for further instructions.).

The models are trained on traceless quadrupoles of the format []
The models trained on the QM9 dataset can be found in the [PiNet2-DQ repository](https://github.com/Teoroo-CMC/PiNet2-DQ)


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


