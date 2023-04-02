from imports import torch, nn

class Loss(nn.Module):
    def forward(self, pred_p, reg_loss, train_m, train_r, dot_scale):
        diff = train_m * (train_r - pred_p)
        sqE = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
        loss_p = sqE + dot_scale*reg_loss
        return loss_p