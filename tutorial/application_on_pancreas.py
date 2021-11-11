import vicdyf
import scvelo as scv
from matplotlib import pyplot as plt
adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
raw_adata = scv.datasets.pancreas()
adata.layers['spliced'] = raw_adata[:, adata.var_names].layers['spliced']
adata.layers['unspliced'] = raw_adata[:, adata.var_names].layers['unspliced']
adata = vicdyf.workflow.estimate_dynamics(adata)
adata = vicdyf.utils.change_visualization(adata, n_neighbors=30)
scv.pl.velocity_embedding_grid(adata,X=adata.obsm['X_vicdyf_umap'], V=adata.obsm['X_vicdyf_mdumap'], color='vicdyf_fluctuation', show=False, basis='X_vicdyf_umap', density=0.3)
plt.savefig('tutorial/pancreas_flow.png')
