# Polarizability Model

The polarizability module PiNet2-$\chi$ implements different models to predict the charge response kernel (CRK)
and polarizability tensor by fitting polarizability tensor data [@Shao_2022]. All models output the polarizability tensor $\boldsymbol{\alpha}$ and CRK $\boldsymbol{\chi}$. The polarizability model requires the dictionary as output from the preprocess layer as input. Listed below are the ``model_params`` that can be set. The EEM [@Mortier85] and ACKS2 [@2013_VerstraelenAyersEtAl] models are based on the
Coulomb kernel and have support for Ewald summation if the Ewald parameters are set and ``cell`` 
is specified in the input data. The EEM and EtaInv models can in addition to polarizability be 
trained on the egap. 

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

## Model specifications

### pinn.models.pol_models.pol_eem_fn
::: pinn.models.pol_models.pol_eem_fn

### pinn.models.pol_models.pol_acks2_fn
::: pinn.models.pol_models.pol_acks2_fn

### pinn.models.pol_models.pol_etainv_fn
::: pinn.models.pol_models.pol_etainv_fn

### pinn.models.pol_models.pol_local_fn
::: pinn.models.pol_models.pol_local_fn

### pinn.models.pol_models.pol_localchi_fn
::: pinn.models.pol_models.pol_localchi_fn

### pinn.models.pol_models.pol_eem_iso_fn
::: pinn.models.pol_models.pol_eem_iso_fn

### pinn.models.pol_models.pol_acks2_iso_fn
::: pinn.models.pol_models.pol_acks2_iso_fn

### pinn.models.pol_models.pol_etainv_iso_fn
::: pinn.models.pol_models.pol_etainv_iso_fn

### pinn.models.pol_models.pol_local_iso_fn
::: pinn.models.pol_models.pol_local_iso_fn

### pinn.models.pol_models.pol_localchi_iso_fn
::: pinn.models.pol_models.pol_localchi_iso_fn

\bibliography