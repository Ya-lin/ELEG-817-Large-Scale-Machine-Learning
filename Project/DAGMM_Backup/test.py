
import numpy as np
import torch
from torch.utils.data import DataLoader


class Valid:
    
    def __init__(self, dagmm):
        self.dagmm = dagmm
    
    def param(self, train_dataset, batch_size):
        dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
        with torch.no_grad():
            N_samples = 0
            gamma_sum = 0
            mu_sum = 0
            cov_sum = 0
            for x, _ in dataloader:
                x = x.float().to(self.dagmm.device)
                _, _, z, gamma = self.dagmm.model(x)
                phi_batch,mu_batch,cov_batch=self.dagmm.compute.compute_params(z,gamma)
                batch_gamma_sum = torch.sum(gamma, dim=0)
                gamma_sum += batch_gamma_sum
                mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
                cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
                N_samples += x.size(0)
                train_phi = gamma_sum / N_samples
                train_mu = mu_sum / gamma_sum.unsqueeze(-1)
                train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
        return train_phi,train_mu,train_cov
    
    def label_energy(self, params, test_data, batch_size):
        phi, mu, cov = params
        dataloader = DataLoader(test_data, batch_size, shuffle=False)
        Pred_Y = np.empty(0)
        for x, y in dataloader:
            _, _, z, gamma = self.dagmm.model(x.to(self.dagmm.device))
            pred, _ = self.dagmm.compute.compute_energy(z, gamma, phi, mu, cov, False)
            pred = pred.detach().cpu().numpy()
            Pred_Y = np.concatenate((Pred_Y, pred), axis=None)
        return Pred_Y

