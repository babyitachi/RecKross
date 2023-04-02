from imports import nn
from kCrossKernelLayer import kCrossKernelLayer
from kernelLayer import KernelLayer

class RecKross(nn.Module):
    def __init__(self, n_u, n_hid, n_dim, n_dep, k_len, n_layers, lambda_s, lambda_2, lambda_kernal, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(1, n_layers):
            if i == 1:
                layers.append(KernelLayer(n_u, n_hid, n_dim, n_dep, lambda_s, lambda_2, inp=True, activation=nn.Sigmoid()))
            elif i%2 == 0:
                layers.append(kCrossKernelLayer(n_u, n_hid, n_dim, n_dep, k_len, lambda_kernal, lambda_kernal, activation=nn.ReLU()))
            else:
                layers.append(KernelLayer(n_hid, n_hid, n_dim, n_dep, lambda_s, lambda_2))
        layers.append(KernelLayer(n_hid, n_u, n_dim, n_dep, lambda_s, lambda_2, activation=nn.Identity(), out=True))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        total_reg = None
        for i, layer in enumerate(self.layers):
            x, reg = layer(x)
            if i < len(self.layers)-1:
                x = self.dropout(x)
            if total_reg is None:
                total_reg = reg
            else:
                total_reg += reg
        return x, total_reg