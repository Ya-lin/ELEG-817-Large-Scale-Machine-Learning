# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:59:40 2022

@author: yalin
"""

#%%
import pandas as pd
from tqdm import trange
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
# import scipy.io

#%% Read Data
Data_NN = pd.read_csv("Data_Correct.csv",header=None)
Data_NN = Data_NN.to_numpy()

#%% (117x90)x240, plot the first dimension
x1 = Data_NN[0,:]
x2 = Data_NN[117,:]
x3 = Data_NN[234,:]
plt.plot(x1,label="m1")
plt.plot(x2,label="m2")
plt.plot(x3,label="m3")
plt.legend()
plt.show()

#%% Understand data
n_dim = 117
T = Data_NN.shape[1]
R = int(Data_NN.shape[0]/n_dim)

#%% transform data in size of (240x90)x117 to perform PCA
# Data consists of 212600 samples with 117 dimensions
Corr_Data = Data_NN[0:n_dim,:].T
for i in trange(1,R):
    indv = Data_NN[i*n_dim:(i+1)*n_dim,:].T
    Corr_Data = np.concatenate((Corr_Data,indv),axis=0)
    
#%% perform PCA, reduced dimension is 3
'''
mean = Corr_Data.mean(0).reshape(1,-1)
Corr_Data_ = Corr_Data-mean
Cov = Corr_Data_.T@Corr_Data_/(Corr_Data_.shape[0]-1)
'''
n_pc = 3
Cov = np.cov(Corr_Data.T)
S,U = LA.eig(Cov)
P = U[:,0:n_pc].real
Corr_Data_Re = Corr_Data@P

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_d = pca.fit_transform(Corr_Data)
