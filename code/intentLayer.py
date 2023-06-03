from imports import nn, torch, F
from localKernal import local_kernel

class IntentLayer(nn.Module):
    def __init__(self, m, n, n_nbr, n_dim, lambda_reg, lambda_reg2, beta):
        super().__init__()
        self.k = n_nbr        
        self.n, self.m = n, m
        self.lambda_reg = lambda_reg
        self.lambda_reg2 = lambda_reg2
        self.beta = beta

        self.ii, self.ii_ind = None, None
        self.uu, self.uu_ind = torch.zeros([n, n]), torch.zeros([n, n])
        
        self.u_hor = nn.Parameter(torch.randn(1, 1, n_dim, n_nbr))
        self.v_hor = nn.Parameter(torch.randn(1, n_nbr, n_dim, n_nbr))
        self.W_hor = nn.Parameter(torch.randn(1, n_nbr, n_nbr))
        
        self.u_ver = nn.Parameter(torch.randn(1, 1, n_dim, n_nbr))
        self.v_ver = nn.Parameter(torch.randn(1, n_nbr, n_dim, n_nbr))
        self.W_ver = nn.Parameter(torch.randn(1, n_nbr, n_nbr))
        
        nn.init.xavier_uniform_(self.u_hor, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v_hor, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.W_hor, gain=torch.nn.init.calculate_gain("relu"))

        nn.init.xavier_uniform_(self.u_ver, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v_ver, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.W_ver, gain=torch.nn.init.calculate_gain("relu"))
        
    def _debias(self, x, min_v, max_v, mean_v):
        x = (torch.abs(x-min_v)+torch.abs(x-max_v))/(torch.log(x))
        return x/(torch.max(x)-torch.min(x))
    
    def _getRatingsProb(self, x):
        x = torch.round(x)
        probs, counts = torch.ones(x.shape[0], 5), torch.ones(x.shape[0])
        for i in range(x.shape[0]):
            ps = torch.histc(x[i], bins=5, min=1., max=5.)
            count = sum(ps)+5.0
            ps = ps+1.0
            ps = ps/count
            counts[i]=count
            probs[i]=ps
        counts = counts/max(counts)
        return probs, counts
    
    def _interaction(self, x, movAvg=False, inter=None):
        probs, counts = self._getRatingsProb(x)
        counts = counts.unsqueeze(1) 
        countsInteraction = torch.matmul(counts, counts.permute(1, 0))
        norms = torch.norm(probs, dim=1).unsqueeze(1)
        interactionMatrix = torch.matmul(probs, probs.permute(1, 0))
        interactionMatrixNorm = torch.matmul(norms, norms.permute(1, 0)) # cosine similarity 
        interactionMatrix = interactionMatrix/interactionMatrixNorm
        interactionMatrix = interactionMatrix*countsInteraction
        if movAvg:
            interactionMatrix = (1-self.beta)*interactionMatrix + self.beta*inter
        interactionMatrixArgSort = torch.argsort(interactionMatrix, dim=1, descending=True)
        return interactionMatrix, interactionMatrixArgSort # cosine similarity, sorted indexes
    
    def _topKusers(self, x):
        users = self.uu_ind[:,:self.k]
        userNBR = torch.stack([x[i] for i in users])
        return userNBR
    
    def _topKitems(self, x):
        items = self.ii_ind[:,:self.k]
        itemNBR = torch.stack([x[i] for i in items])
        return itemNBR
    
    def _aggregate(self, x):
        userNBRs = self._topKusers(x.permute(1,0)).double()
        itemNBRs = self._topKitems(x).double()
        
        userKrnl = (self.W_ver * local_kernel(self.u_ver, self.v_ver)).double()
        itemKrnl = (self.W_ver * local_kernel(self.u_hor, self.v_hor)).double()

        sparse_reg_userKrnl = self.lambda_reg * torch.nn.functional.mse_loss(userKrnl, torch.zeros_like(userKrnl))
        sparse_reg_itemKrnl = self.lambda_reg * torch.nn.functional.mse_loss(itemKrnl, torch.zeros_like(itemKrnl))
        
        l2_reg_user = self.lambda_reg2 * torch.nn.functional.mse_loss(self.W_ver, torch.zeros_like(self.W_ver))        
        l2_reg_item = self.lambda_reg2 * torch.nn.functional.mse_loss(self.W_hor, torch.zeros_like(self.W_hor))        
        
        u_conv = nn.functional.conv1d(userNBRs, userKrnl, padding='same').squeeze(1)
        i_conv = nn.functional.conv1d(itemNBRs, itemKrnl, padding='same').squeeze(1)
        return u_conv.permute(1,0), i_conv, sparse_reg_userKrnl + sparse_reg_itemKrnl + l2_reg_user + l2_reg_item
    
    def _updateProbs(self, x):
        # taking moving average of the probabilities
        self.ii, self.ii_ind = self._interaction(x, movAvg=True, inter=self.ii)
        self.uu, self.uu_ind = self._interaction(x.permute(1, 0), movAvg=True, inter=self.uu)
    
    def forward(self, x):
        if self.ii is None:
            self.ii, self.ii_ind = self._interaction(x)
            self.uu, self.uu_ind = self._interaction(x.permute(1, 0))
        
        u, i, reg = self._aggregate(x)
        x = torch.pow(x + u + i, 3)
        return x, reg