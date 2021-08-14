import vicdyf
import scvelo as scv
adata = scv.datasets.pancreas()
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=4000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
raw_adata = scv.datasets.pancreas()
adata.layers['spliced'] = raw_adata[:, adata.var_names].layers['spliced']
adata.layers['unspliced'] = raw_adata[:, adata.var_names].layers['unspliced']
adata = vicdyf.workflow.estimate_dynamics(adata)
