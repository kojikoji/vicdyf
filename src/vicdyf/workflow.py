import scanpy as sc
import torch
from . import dataset, modules, utils


def estimate_dynamics(
        adata, use_genes=None, first_epoch=500, second_epoch=500, param_path='.vicdyf_opt.pt',
        model_params = {
            'z_dim': 10,
            'enc_z_h_dim': 50, 'enc_d_h_dim': 50, 'dec_z_h_dim': 50,
            'num_enc_z_layers': 2, 'num_enc_d_layers': 2,
            'num_dec_z_layers': 2    
        },
        lr=0.001, val_ratio=0, test_ratio=0.05,
        batch_size=100, num_workers=1, sample_num=10,
        n_neighbors=30, min_dist=0.1):
    if use_genes == None:
        use_genes = adata.var_names
    utils.input_checks(adata)
    adata.var['vicdyf_used'] = use_genes
    vicdyf_exp = utils.define_exp(
        adata,
        model_params= model_params,
        lr=lr, val_ratio=val_ratio, test_ratio=test_ratio,
        batch_size=batch_size, num_workers=num_workers)    
    vicdyf_exp.model.no_d_kld = True
    vicdyf_exp.model.no_lu = True
    print(f'Loss:{vicdyf_exp.test()}')
    print('Start first opt')
    for param in vicdyf_exp.model.enc_d.parameters():
        param.requires_grad = False
    vicdyf_exp.init_optimizer(lr)
    vicdyf_exp.train_total(first_epoch)
    print('Done first opt')
    print(f'Loss:{vicdyf_exp.test()}')
    print('Start second opt')
    vicdyf_exp.model.no_lu = False
    vicdyf_exp.model.no_d_kld = False
    for param in vicdyf_exp.model.enc_d.parameters():
        param.requires_grad = True
    vicdyf_exp.init_optimizer(lr)
    vicdyf_exp.train_total(second_epoch)
    print('Done second opt')
    print(f'Loss:{vicdyf_exp.test()}')
    torch.save(vicdyf_exp.model.state_dict(), param_path)
    adata.uns['param_path'] = param_path
    adata = utils.post_process(adata, vicdyf_exp, sample_num=sample_num, n_neighbors=n_neighbors, min_dist=min_dist)
    return(adata)
