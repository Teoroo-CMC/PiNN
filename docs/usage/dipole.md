# Dipole Model

The dipole model requires the same dictionary as input as the potential model.
The only difference is the ``model_params`` that can be set. They are listed below
along with their default values.

| Parameter           | Default | Description                                                                      |
|---------------------|---------|----------------------------------------------------------------------------------|
| `max_dipole`        | `False` | When set to a number, exclude dipoles above this value in loss function          |
| `d_scale`           | `1`     | Dipole unit during training                                                      |
| `use_d_per_atom`    | `False` | Use the per-atom dipole in place of total dipole in loss function                |
| `log_d_per_atom`    | `True`  | Log the per-atom dipole error, automatically enabled with `use_d_per_atoms=True` |
| `use_d_weight`      | `False` | Scale the energy loss according to the `'d_weight'` Tensor in the dataset        |
| `d_loss_multiplier` | 1       | Weight of dipole loss                                                            |
