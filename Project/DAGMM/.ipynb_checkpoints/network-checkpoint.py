
import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        """
        Network for DAGMM on Rehabilitation exercise dataset (Idaho)
        n_gmm: number of components of GMM
        z_dim: reduced dimension
        2 and 1 are the values
        """
        
        super(DAGMM, self).__init__()
        #Encoder network
        self.en1 = nn.LSTM(117,30,batch_first=True)
        self.en2 = nn.LSTM(30,10,batch_first=True)
        self.en3 = nn.LSTM(10,z_dim,batch_first=True)

        #Decoder network
        self.de1 = nn.LSTM(z_dim,10,batch_first=True)
        self.de2 = nn.LSTM(10,30,batch_first=True)
        self.de3 = nn.LSTM(30,117,batch_first=True)

        #Estimation network
        self.et1 = nn.Linear(z_dim+2, 10)
        self.et2 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h,(_,_) = self.en1(x)
        h,(_,_) = self.en2(h)
        h,(_,_) = self.en3(h)
        return h

    def decode(self, x):
        h,(_,_) = self.de1(x)
        h,(_,_) = self.de2(h)
        h,(_,_) = self.de3(h)
        return h
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.et1(z)), 0.5)
        return F.softmax(self.et2(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma

