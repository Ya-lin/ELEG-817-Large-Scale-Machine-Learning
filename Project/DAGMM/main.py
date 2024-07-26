# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:57:59 2024

@author: yalin
"""

from pathlib import Path
import numpy as np
import torch

import matplotlib.pyplot as plt
import pandas as pd 

from train import TrainerDAGMM


#%%
class Args:
    def __init__(self):
        self.num_epochs = 200
        self.patience = 50
        self.lr = 1e-4
        self.lr_milestones = [50]
        self.latent_dim = 2
        self.n_gmm = 6
        self.lambda_energy = 0.1
        self.lambda_cov = 0.005
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()
print(vars(args))


#%%
path = Path.home().joinpath("Documents","Data","Rehabilitation")
path.mkdir(exist_ok=True)
def get_data():
    X = pd.read_csv(path.joinpath("Data_Correct.csv"),header=None)
    Y = pd.read_csv(path.joinpath("Data_Incorrect.csv"),header=None)
    X = X.to_numpy()
    Y = Y.to_numpy()
    return X, Y

X, Y = get_data()
print(X.shape)
print(Y.shape)


#%%
n_dim = 117
T = X.shape[1]
R = int(X.shape[0]/n_dim)
def two2three(X, R, T, n_dim):
    X_NN = np.zeros((R,T,n_dim))
    for r in range(R):
        X_NN[r,:,:] = X[r*n_dim:(r+1)*n_dim,:].T
    X_NN = torch.tensor(X_NN).float()
    return X_NN

X_NN = two2three(X, R, T, n_dim)
Y_NN = two2three(Y, R, T, n_dim)
print(X_NN.shape)
print(Y_NN.shape)


#%%
X_NN = X_NN.reshape((-1, 117))

dagmm = TrainerDAGMM(args, X_NN)

z_c, x_hat, z, gamma = dagmm.model(X_NN.to(args.device))

print(z_c.shape)
print(x_hat.shape)




