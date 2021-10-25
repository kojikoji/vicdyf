import os
import sys
import pandas as pd
import numpy as np
import umap
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
import torch.distributions as dist
import torch
from . import vicdyf


def embed_z(z_mat, n_neighbors=30, min_dist=0.3):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    z_embed = reducer.fit_transform(z_mat)
    return(z_embed)
    

def input_checks(adata):
    if (not 'spliced' in adata.layers.keys()) or (not 'unspliced' in adata.layers.keys()):
        raise ValueError(
            f'Input anndata object need to have layers named `spliced` and `unspliced`.')
    if np.sum((adata.layers['spliced'] - adata.layers['spliced'].astype(int)))**2 != 0:
        raise ValueError('layers `spliced` includes non integer number, while count data is required for `spliced`.')
    
    if np.sum((adata.layers['unspliced'] - adata.layers['unspliced'].astype(int)))**2 != 0:
        raise ValueError('layers `unspliced` includes non integer number, while count data is required for `unspliced`.')
    
    return(adata)

def define_exp(
        adata,
        model_params = {
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 50, 'enc_d_h_dim': 50, 'dec_z_h_dim': 50,
            'num_enc_z_layers': 2, 'num_enc_d_layers': 2,
            'num_dec_z_layers': 2    
        },
        lr=0.0001, val_ratio=0.05, test_ratio=0.1,
        batch_size=300, num_workers=1):
    # splice
    select_adata = adata[:, adata.var['vicdyf_used']]
    if type(select_adata.layers['spliced']) == np.ndarray:
        s = torch.tensor(select_adata.layers['spliced'])
    else:
        s = torch.tensor(select_adata.layers['spliced'].toarray())
    # unspliced
    if type(select_adata.layers['unspliced']) == np.ndarray:
        u = torch.tensor(select_adata.layers['unspliced'])
    else:
        u = torch.tensor(select_adata.layers['unspliced'].toarray())
    # meta df
    model_params['x_dim'] = s.shape[1]
    vicdyf_exp = vicdyf.VicDyfExperiment(model_params, lr, s, u, test_ratio, batch_size, num_workers, validation_ratio=val_ratio)
    return(vicdyf_exp)


    

def calc_tr_mat(z, d, sigma):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    zz_diff_mat = z.view(1, cell_num, gene_num) - z.view(cell_num, 1, gene_num)
    norm_fix_mat = (zz_diff_mat == 0).type(torch.DoubleTensor)
    zz_diff_mat /= torch.norm(zz_diff_mat, dim=2, p=2, keepdim=True) + norm_fix_mat
    d = d.view(cell_num, 1, gene_num)
    d /= torch.norm(d, dim=2, p=2, keepdim=True)
    cos_sim = torch.sum(d * zz_diff_mat, dim=2)
    tr_mat = torch.exp(cos_sim / sigma)
    tr_mat /= torch.sum(tr_mat, axis=1).view(-1, 1)
    tr_mat = (tr_mat - 1/cell_num)
    return(tr_mat)

def embed_tr_mat(z_embed, tr_mat, gene_norm):
    z_embed = torch.tensor(z_embed)
    cell_num = z_embed.shape[0]
    gene_num = z_embed.shape[1]
    zz_diff_mat = z_embed.view(1, cell_num, gene_num) - z_embed.view(cell_num, 1, gene_num)
    norm_fix_mat = (zz_diff_mat == 0).type(torch.DoubleTensor)
    zz_diff_mat /= torch.norm(zz_diff_mat, dim=2, p=2, keepdim=True) + norm_fix_mat
    d_embed = torch.sum(tr_mat.view(cell_num, cell_num, 1) * zz_diff_mat, dim=1).view(cell_num, 2)
    d_norm = np.linalg.norm(d_embed, axis=1)
    d_embed *= (gene_norm.reshape(-1, 1) / d_norm.reshape(-1, 1))
    return(d_embed)
    
def calc_gene_mean_sd(z, qd, dcoeff, model, sample_num=50):
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    zd_batch = z.view(1, cell_num, gene_num) + dcoeff * qd.sample((sample_num,))
    px_ld_batch = model.dec_z(zd_batch) - model.dec_z(z).view(1, cell_num, -1)
    batch_std_mat = torch.std(px_ld_batch, dim=0)
    batch_mean_mat = torch.mean(px_ld_batch, dim=0)
    gene_sd = torch.norm(batch_std_mat, dim=1).cpu().detach().numpy()
    gene_mean = torch.norm(batch_mean_mat, dim=1).cpu().detach().numpy()
    return(gene_mean, gene_sd, batch_mean_mat, batch_std_mat)
    

def post_process(adata, vicdyf_exp, sigma=0.05, n_neighbors=30, min_dist=0.1, dz_var_prop=0.05, sample_num=10):
    x = vicdyf_exp.edm.s
    u = vicdyf_exp.edm.u
    vicdyf_exp.device = torch.device('cpu')
    vicdyf_exp.model = vicdyf_exp.model.to(vicdyf_exp.device)
    z, d, qz, qd, px_z_ld, pu_zd_ld = vicdyf_exp.model(x)
    # make zl centered embedding
    zl = qz.loc
    qd_mu, qd_logvar = vicdyf_exp.model.enc_d(zl)
    qd = dist.Normal(qd_mu, vicdyf_exp.model.softplus(qd_logvar))
    d = vicdyf_exp.model.d_coeff * qd.rsample()
    px_z_ld = vicdyf_exp.model.dec_z(zl)
    model_d_coeff = vicdyf_exp.model.d_coeff
    dl = qd.loc * model_d_coeff
    cell_num = z.shape[0]
    gene_num = z.shape[1]
    # store basic stats
    d_mat = np.copy(d.cpu().detach().numpy())
    z_mat = np.copy(z.cpu().detach().numpy())
    zl_mat = np.copy(zl.cpu().detach().numpy())
    dl_mat = np.copy(dl.cpu().detach().numpy())
    ld_mat = np.copy(px_z_ld.cpu().detach().numpy())
    adata.obsm['X_vicdyf_z'] = z_mat
    adata.obsm['X_vicdyf_zl'] = zl_mat
    adata.obsm['X_vicdyf_d'] = d_mat
    adata.obsm['X_vicdyf_dl'] = dl_mat
    print('Extract info')
    # calc gene velocity and fluctiona
    pxd_zd_ld = vicdyf_exp.model.dec_z(zl + d)
    gene_vel = (pxd_zd_ld - px_z_ld).cpu().detach().numpy()
    gene_norm = np.linalg.norm(gene_vel, axis=1)
    mean_gene_norm, gene_sd, mean_gene_vel, batch_std_mat = calc_gene_mean_sd(zl, qd, model_d_coeff, vicdyf_exp.model, sample_num=sample_num)
    batch_std_mat = batch_std_mat.cpu().detach().numpy()
    mean_gene_vel = mean_gene_vel.cpu().detach().numpy()
    adata.layers['vicdyf_expression'] = px_z_ld.cpu().detach().numpy()
    adata.layers['vicdyf_velocity'] = gene_vel
    adata.layers['vicdyf_mean_velocity'] = mean_gene_vel
    adata.layers['vicdyf_fluctuation'] = batch_std_mat
    adata.obs['vicdyf_fluctuation'] = np.mean(adata.layers['vicdyf_fluctuation'], axis=1)
    adata.obs['vicdyf_velocity'] = np.mean(np.abs(adata.layers['vicdyf_velocity']), axis=1)
    adata.obs['vicdyf_mean_velocity'] = np.mean(np.abs(adata.layers['vicdyf_mean_velocity']), axis=1)
    # calculate transition rate
    stoc_tr_mat = calc_tr_mat(zl.cpu().detach(), d.cpu().detach(), sigma)
    mean_tr_mat = calc_tr_mat(zl.cpu().detach(), dl.cpu().detach(), sigma)
    # embed z
    z_embed = embed_z(zl_mat, n_neighbors=n_neighbors, min_dist=min_dist)
    adata.obsm['X_vicdyf_umap'] = z_embed
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, adata.obs['vicdyf_velocity'].values)
    mean_d_embed =embed_tr_mat(z_embed, mean_tr_mat, adata.obs['vicdyf_mean_velocity'].values)
    adata.obsp['stoc_tr_mat'] = stoc_tr_mat.detach().numpy()
    adata.obsp['mean_tr_mat'] = mean_tr_mat.detach().numpy()
    adata.obsm['X_vicdyf_sdumap'] = stoc_d_embed.cpu().detach().numpy()
    adata.obsm['X_vicdyf_mdumap'] = mean_d_embed.cpu().detach().numpy()
    return(adata)

def change_visualization(adata, embeddings=None, n_neighbors=30, min_dist=0.1):
    # embed z
    if embeddings == None:
        z_embed = embed_z(adata.obsm['X_vicdyf_zl'], n_neighbors=n_neighbors, min_dist=min_dist)
    else:
        if type(embeddings) == str:
            z_embed = adata.obsm[embeddings]
        else:
            z_embed = embeddings
    adata.obsm['X_vicdyf_umap'] = z_embed
    stoc_tr_mat = torch.tensor(adata.obsp['stoc_tr_mat'])
    mean_tr_mat = torch.tensor(adata.obsp['mean_tr_mat'])
    stoc_d_embed = embed_tr_mat(z_embed, stoc_tr_mat, adata.obs['vicdyf_velocity'].values)
    mean_d_embed =embed_tr_mat(z_embed, mean_tr_mat, adata.obs['vicdyf_mean_velocity'].values)
    adata.obsm['X_vicdyf_sdumap'] = stoc_d_embed.cpu().detach().numpy()
    adata.obsm['X_vicdyf_mdumap'] = mean_d_embed.cpu().detach().numpy()
    return(adata)
    
