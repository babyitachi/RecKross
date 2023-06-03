from imports import nn
from intentLayer import IntentLayer
from kernelLayer import KernelLayer

class IntentRec(nn.Module):
    def __init__(self, n_m, n_u, n_hid, n_dim, k, lambda_s, lambda_2, lambda_reg, lambda_reg2, beta, n_layers, dropout=0.5):
        super().__init__()
        layers = []
        self.oldReg = 0.0
        layers.append(IntentLayer(n_m, n_u, k, n_hid, lambda_reg, lambda_reg2, beta))
        for i in range(1, n_layers):
                if i==1:
                    layers.append(KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2, inp=True, activation=nn.Sigmoid()))
                else:
                    layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2, inp=True, activation=nn.Sigmoid()))
        layers.append(KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity(), out=True))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def _updateProbs(self, x, reg):
        if self.oldReg==0.0:
            self.oldReg = reg
            return
        if round(reg, 2)<round(self.oldReg, 2):
            print(self.oldReg, reg)
            self.layers[0]._updateProbs(x)
        self.oldReg = (self.oldReg + reg)/2
            
    def forward(self, x):
        total_reg = None
        for i, layer in enumerate(self.layers):
            x, reg = layer(x)
            x = x.double()
            if i < len(self.layers)-1:
                x = self.dropout(x)
            if total_reg is None:
                total_reg = reg
            else:
                total_reg += reg
        return x, total_reg