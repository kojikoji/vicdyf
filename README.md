# vicdyf: Variational Inference of Cell state Dynamics with fluctuation
vicdyf is intended to estimated cell state dynamics with fluctuation from spliced and unspliced transcript abundance.

## Instalation
You can install vicdyf using pip command from your shell.
```shell
pip install vicdyf
```

## Usage
You need to prepare an [`AnnData` object](https://anndata.readthedocs.io/en/latest/) which includes raw spliced and unspliced counts as `layers` named  as `spliced` and `unspliced` like a [scvelo data set](https://scvelo.readthedocs.io/Pancreas/). Apply `vicdyf` workflow on the object:

```python
import vicdyf
adata = vicdyf.workflow.estimate_dynamics(adata)
```

`vicdyf.workflow.estimate_dynamics` have optional parameters as below:
- `use_genes`: gene names for dynamics estimation (default: `None`)
- `first_epoch`: number of epochs for deriving latent representation (default: `500`)
- `second_epoch`: number of epochs for optimizing dynamics (default: `500`)
- `param_path`: a path where the optimized parameters of `vicdyf.modules.VicDyf`are stored (default: `.vicdyf_opt_pt`)
- `lr`: Learning rate for [Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) of [pytorch](https://pytorch.org)
- `batch_size`: Size of mini batches in the optimization procedure
- `num_workers`: Number of workers in [data loader](https://pytorch.org/docs/stable/data.html) of [pytorch](https://pytorch.org)
- `val_ratio`: proportion of validation data set
- `test_ratio`: proportion of test data set
- `model_params`: a dictionary which describe the configuration of `vicdyf.modules.VicDyf`. The keys of the dictionary is as below:
  - `z_dim`: dimension of latent representation (default `10`)
  - `enc_z_h_dim`: dimension of hidden units in encoder layers (default `50`)
  - `enc_d_h_dim`: dimension of hidden units in dynamics encoder layers (default `50`)
  - `dec_z_h_dim`: dimension of hidden units in encoder layers (default `50`)
  - `num_enc_z_layers`: the layer number of the encoder (default `2`)
  - `num_enc_z_layers`: the layer number of the dynamics encoder (default `2`)
  - `num_dec_z_layers`: the layer number of the decoder (default `2`)


Here, the `AnnData` object acuires sevral elements in `layers`, `obsm`, `obsp` and `obs`.
- `layers`: 
  - `vicdyf_expression`: Expected gene expression level
  - `vicdyf_mean_velocity`: Expected gene expression change
  - `vicdyf_velocity`: Stochasticaly sampled  gene expression change
  - `vicdyf_fluctuation`: Fluctuation level for each gene
- `obsm`:
  - `X_vicdyf_z`: Stochasticaly smapled latent representation
  - `X_vicdyf_zl`: Expected latent representation
  - `X_vicdyf_d`: Stochasticaly smapled changes of latent representation
  - `X_vicdyf_dl`: Expected changes of latent representation
  - `X_vicdyf_umap`: 2D UMAP embeddings of expected latent representation for visualization
  - `X_vicdyf_sdumap`: 2D UMAP embeddings of `X_vicdyf_d` for visualization
  - `X_vicdyf_mdumap`: 2D UMAP embeddings of `X_vicdyf_dl` for visualization

