# evndyn
import torch
from .modules import VicDyf
from .dataset import VicDyfDataSet, VicDyfDataManager
from torch.utils.data import DataLoader
import numpy as np
import copy

class VicDyfExperiment:
    def __init__(self, model_params, lr, s, u, test_ratio, batch_size, num_workers, validation_ratio=0.1):
        self.edm = VicDyfDataManager(s, u, test_ratio, batch_size, num_workers, validation_ratio=validation_ratio)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VicDyf(**model_params)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_loss = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        entry_num = 0
        for s, u, norm_mat in self.edm.train_loader:
            s = s.to(self.device)
            u = u.to(self.device)
            norm_mat = norm_mat.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.elbo_loss(
                s, u, norm_mat)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            entry_num += s.shape[0]
        return(total_loss / entry_num)

    def evaluate(self):
        self.model.eval()
        s = self.edm.validation_s.to(self.device)
        u = self.edm.validation_u.to(self.device)
        norm_mat = self.edm.validation_norm_mat.to(self.device)
        loss = self.model.elbo_loss(
            s, u, norm_mat)
        entry_num = s.shape[0]
        loss_val = loss / entry_num
        return(loss_val)

    def test(self):
        self.model.eval()
        s = self.edm.test_s.to(self.device)
        u = self.edm.test_u.to(self.device)
        norm_mat = self.edm.test_norm_mat.to(self.device)
        loss = self.model.elbo_loss(
            s, u, norm_mat)
        entry_num = s.shape[0]
        loss_val = loss / entry_num
        return(loss_val)

    def train_total(self, epoch_num):
        for epoch in range(epoch_num):
            state_dict = copy.deepcopy(self.model.state_dict())
            loss = self.train_epoch()
            if np.isnan(loss):
                self.model.load_state_dict(state_dict)
                break
            if epoch % 10 == 0:
                print(f'loss at epoch {epoch} is {loss}')

    def get_forward(self, s):
        z, d, qz, qd, px_z_ld, pu_zd_ld = self.model(s)
        return(z, d, qz, qd, px_z_ld, pu_zd_ld)

    def init_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
