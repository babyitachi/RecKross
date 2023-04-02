from imports import nn, torch, F
from localKernal import local_kernel

class kCrossKernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, n_dep, k_len, lambda_hor, lambda_ver, activation=nn.Sigmoid()):
        super().__init__()
        
        self.u_hor = nn.Parameter(torch.randn(n_dep, 1, 1, n_hid, n_dim))
        self.v_hor = nn.Parameter(torch.randn(n_dep, 1, k_len, n_hid, n_dim))
        
        self.u_ver = nn.Parameter(torch.randn(n_dep, 1, 1, n_hid, n_dim))
        self.v_ver = nn.Parameter(torch.randn(n_dep, 1, k_len, n_hid, n_dim))
        
        self.b_hor = nn.Parameter(torch.randn(n_dep, 1, n_hid))        
        self.b_ver = nn.Parameter(torch.randn(n_dep, 1, n_hid))
        
        self.lambda_hor = lambda_hor
        self.lambda_ver = lambda_ver
        
        nn.init.xavier_uniform_(self.u_hor, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v_hor, gain=torch.nn.init.calculate_gain("relu"))

        nn.init.xavier_uniform_(self.u_ver, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v_ver, gain=torch.nn.init.calculate_gain("relu"))
        
        nn.init.zeros_(self.b_hor)
        nn.init.zeros_(self.b_ver)
        self.activation = activation
        
    def forward(self, x):
        k_hor = local_kernel(self.u_hor, self.v_hor)
        k_ver = local_kernel(self.u_ver, self.v_ver)

        sparse_reg_hor = torch.nn.functional.mse_loss(k_hor, torch.zeros_like(k_hor))
        sparse_reg_term_hor = self.lambda_hor * sparse_reg_hor
        
        sparse_reg_ver = torch.nn.functional.mse_loss(k_ver, torch.zeros_like(k_ver))
        sparse_reg_term_ver = self.lambda_ver * sparse_reg_ver
        
        inp=x
        
        x = x.unsqueeze(1)
        hor = F.conv2d(x, k_hor.double(), padding='same')
        hor = torch.mean(hor, 1)
        hor += self.b_hor        
        
        ver = F.conv2d(x.permute(0, 1, 3, 2), k_ver.double(), padding='same')
        ver = torch.mean(ver, 1).permute(0, 2, 1)
        ver += self.b_hor
        
        y = self.activation(inp + hor + ver)
        
        return y, sparse_reg_term_hor + sparse_reg_term_ver