from imports import nn, torch 
from localKernal import local_kernel

class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_in, n_hid))
        self.u = nn.Parameter(torch.randn(n_in, 1, n_dim))
        self.v = nn.Parameter(torch.randn(1, n_hid, n_dim))
        self.b = nn.Parameter(torch.randn(1, n_hid))

        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2

        nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.b)
        self.activation = activation

    def forward(self, x):
        w_hat = local_kernel(self.u, self.v)
        sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
        sparse_reg_term = self.lambda_s * sparse_reg

        l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
        l2_reg_term = self.lambda_2 * l2_reg
        
        W_eff = self.W * w_hat  # Reparameterizing Local kernelised weight matrix
        y = torch.matmul(x, W_eff.double()) + self.b.double()
        y = self.activation(y)
        return y, sparse_reg_term + l2_reg_term