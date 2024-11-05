# Polarizability Model

The polarizability model requires the same dictionary as input as the potential model.
The only difference is the ``model_params`` that can be set. They are listed below
along with their default values.

| Parameter            | Default | Description                                                                      |
|----------------------|---------|----------------------------------------------------------------------------------|
| `ewald_rc`         | `None` | Ewald short-range cut-off          |
| `ewald_kmax`            | `None`     | Maximum k for Ewald summation                                                     |
| `ewald_eta`     | `None` | Gaussian width for Ewald summation               |
| `p_scale`     | `1`  | Polarization unit during training |
| `p_unit`       | `1` | Output unit of polarizability during prediction (default: atomic units)       |                                             |
| `p_loss_multiplier`  | `1`     | Weight of polarizability loss                                                            |
| `train_egap` | `0`     | Whether to train on egap data                                                                      |
| `eval_egap` | `0`     | Whether to return egap predictions                                                                    |
