import scanpy as sc
import torch
from . import dataset, modules, utils


def estimate_dynamics(adata, use_genes=None, first_epoch=500, second_epoch=500, param_path='.vicdyf_opt.pt'):
    if len(use_genes) == 0:
        use_genes = adata.var_names
    utils.input_checks(adata)
    adata.var['vicdyf_used'] = use_genes
    vicdyf_exp = utils.define_exp(adata)
    vicdyf_exp.model.no_d_kld = True
    vicdyf_exp.model.no_lu = True
    print(f'Loss:{vicdyf_exp.test()}')
    print('Start first opt')
    for param in vicdyf_exp.model.enc_d.parameters():
        param.requires_grad = False
    vicdyf_exp.train_total(first_epoch)
    print('Done first opt')
    print(f'Loss:{vicdyf_exp.test()}')
    print('Start second opt')
    vicdyf_exp.model.no_lu = False
    vicdyf_exp.model.no_d_kld = False
    for param in vicdyf_exp.model.enc_d.parameters():
        param.requires_grad = True
    vicdyf_exp.train_total(second_epoch)
    print('Done second opt')
    print(f'Loss:{vicdyf_exp.test()}')
    torch.save(vicdyf_exp.model.state_dict(), param_path)
    adata.uns['param_path'] = param_path
    adata = utils.post_process(adata, vicdyf_exp)
    return(adata)
