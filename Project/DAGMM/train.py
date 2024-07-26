
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from network import DAGMM


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.linalg.inv(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5*torch.sum(torch.sum(z_mu.unsqueeze(-1)*cov_inverse.unsqueeze(0), dim=-2)*z_mu,                                        dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z/(torch.sqrt(det_cov)).unsqueeze(0), dim=1)+eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        

class Cholesky(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a):
        l = torch.linalg.cholesky(a, upper=False)
        ctx.save_for_backward(l)
        return l
    
    @staticmethod
    def backward(ctx, grad_output):
        l, = ctx.saved_tensors
        linv = torch.linalg.inv(l)
        inner = torch.tril(torch.mm(l.t(), grad_output))*torch.tril(
                           1.0 - torch.diag(torch.full((l.size(1),), 0.5)))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class TrainerDAGMM:
    """Trainer class for DAGMM."""
    
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(args.device)
        self.model.apply(weights_init_normal)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   args.device, self.args.n_gmm)
    def train(self):
        """Training the DAGMM model"""
        
        self.model.train()
        for epoch in range(self.args.num_epochs):
            x = self.data.to(self.args.device)
            self.optimizer.zero_grad()     
            _, x_hat, z, gamma = self.model(x)
            loss = self.compute.forward(x, x_hat, z, gamma)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            optimizer.step()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, loss.item()))


# set the way to initialize the model        
def weights_init_normal(m):
    
    classname = m.__class__.__name__
    
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
        
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    

