from time import time
import json
from argparse import ArgumentParser
import h5py as h5py
from scipy.sparse import csc_matrix
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
seed=1234
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")