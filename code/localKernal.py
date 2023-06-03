from imports import torch

def local_kernel(u, v):
    dist = torch.norm(u - v, p=3, dim=3)
    hat = torch.clamp(1. - dist**2, min=0.)
    return hat
