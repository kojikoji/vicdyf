import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np
from .funcs import calc_kld, calc_nb_loss, calc_poisson_loss

class LinearReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.ReLU(True))

    def forward(self, x):
        h = self.f(x)
        return(h)


class SeqNN(nn.Module):
    def __init__(self, num_steps, dim):
        super(SeqNN, self).__init__()
        modules = [
            LinearReLU(dim, dim)
            for _ in range(num_steps)
        ]
        self.f = nn.Sequential(*modules)

    def forward(self, pre_h):
        post_h = self.f(pre_h)
        return(post_h)


class Encoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()
        self.x2h = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        return(mu, logvar)


class Decoder(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        self.z2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        pre_h = self.z2h(z)
        post_h = self.seq_nn(pre_h)
        ld = self.h2ld(post_h)
        correct_ld = self.softplus(ld)
        return(correct_ld)


class VicDyf(nn.Module):
    def __init__(
            self,
            x_dim, z_dim,
            enc_z_h_dim, enc_d_h_dim, dec_z_h_dim,
            num_enc_z_layers, num_enc_d_layers,
            num_dec_z_layers, norm_input=False):
        super(VicDyf, self).__init__()
        self.enc_z = Encoder(num_enc_z_layers, x_dim, enc_z_h_dim, z_dim)
        self.enc_d = Encoder(num_enc_d_layers, z_dim, enc_d_h_dim, z_dim)
        self.dec_z = Decoder(num_enc_z_layers, z_dim, dec_z_h_dim, x_dim)
        self.dt = 1
        self.gamma_mean = 0.05
        self.d_coeff = 0.01
        self.loggamma = Parameter(torch.Tensor(x_dim))
        self.logbeta = Parameter(torch.Tensor(x_dim))
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        self.no_lu = False
        self.no_d_kld = False
        self.no_z_kld = False
        self.norm_input = norm_input

    def reset_parameters(self):
        init.normal_(self.loggamma)
        init.normal_(self.logbeta)
        
    def forward(self, x):
        # encode z
        qz_mu, qz_logvar = self.enc_z(x)
        qz = dist.Normal(qz_mu, self.softplus(qz_logvar))
        z = qz.rsample()
        # encode d
        qd_mu, qd_logvar = self.enc_d(z)
        qd = dist.Normal(qd_mu, self.softplus(qd_logvar))
        d = self.d_coeff * qd.rsample()
        # decode z
        px_z_ld = self.dec_z(z)
        # deconst z + d
        pxd_zd_ld = self.dec_z(z + d)
        pxmd_zd_ld = self.dec_z(z - d)
        diff_px_zd_ld = pxd_zd_ld - pxmd_zd_ld
        gamma = self.softplus(self.loggamma)
        beta = self.softplus(self.logbeta) * self.dt
        pu_zd_ld = self.softplus(diff_px_zd_ld + px_z_ld * gamma) / beta
        return(z, d, qz, qd, px_z_ld, pu_zd_ld)

    def elbo_loss(self, x, u, norm_mat, turn_on_d_kld=False):
        if self.norm_input:
            in_x = x / norm_mat
        else:
            in_x = x
        z, d, qz, qd, px_z_ld, pu_zd_ld = self(in_x)
        # kld of pz and qz
        z_kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
        if self.no_d_kld:
            d_kld = 0
        else:
            d_kld = -0.5 * (1 + qd.scale.pow(2).log() - qd.loc.pow(2) - qd.scale.pow(2))
        if self.no_z_kld:
            z_kld = torch.zeros_like(z_kld)
        kld = z_kld + d_kld
        # reconst loss of x
        lx = calc_poisson_loss(px_z_ld, norm_mat, x)
        # reconst loss of unspliced x
        if self.no_lu:
            lu = 0
        else:
            lu = calc_poisson_loss(pu_zd_ld, norm_mat, u)
        elbo_loss = torch.sum((torch.sum(kld, dim=-1) + torch.sum(lx + lu, dim=-1)))
        return(elbo_loss)
