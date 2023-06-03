from imports import torch

def getOptimizer(params, optim_type, lr=1e-3):
    print(f"Optimizer: {optim_type}")
    if optim_type=='Adam':
        return torch.optim.Adam(params, lr=lr)
    elif optim_type=='LBFGS':
        return torch.optim.LBFGS(params, lr=lr)
    elif optim_type=='AdamW':
        return torch.optim.AdamW(params, lr=lr)
    else:
        print("Invalid Optimizer choice. using AdamW")
        return torch.optim.AdamW(params, lr=lr)


def getConfig(dataset=""):
    conf_ML_1M={
        "dataset": "ML-1M", # datset name; options( "ML-1M", "ML-100K", "Douban")
        "n_hid" : 500, #500 # size of hidden layers
        "n_dep" : 5, # depth on hidden layers
        "n_dim" : 5, # inner AE embedding size
        "k_len" : 3, # length of kernal in NeighbourLayer 
        "n_layers" : 3, # number of hidden layers

        # Hyperparameters to tune for specific case
        "max_epoch_p" : 200, # max number of epochs for pretraining
        "patience_p" : 10, #5 # number of consecutive rounds of early stopping condition before actual stop for pretraining
        "tol_p" : 1e-4, # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
        "lambda_2" : 44., # regularisation of number or parameters
        "lambda_s" : 0.0006291, # regularisation of sparsity of the final matrix
        "lambda_kernal" : 0.03046,
        "lr":0.98,
        "optim_type":'LBFGS',
        "drop": 0.65,
        "dot_scale": 0.351}

    conf_ML_100K={
        "dataset": "ML-100K", # datset name; options( "ML-1M", "ML-100K", "Douban")
        "n_hid" : 500, #500 # size of hidden layers
        "n_dep" : 3, # depth on hidden layers
        "n_dim" : 5, # inner AE embedding size
        "k_len" : 3, # length of kernal in NeighbourLayer 
        "n_layers" : 3, # number of hidden layers

        # Hyperparameters to tune for specific case
        "max_epoch_p" : 5000, # max number of epochs for pretraining
        "patience_p" : 50, #5 # number of consecutive rounds of early stopping condition before actual stop for pretraining
        "tol_p" : 1e-4, # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
        "lambda_2" : 22., # regularisation of number or parameters
        "lambda_s" : 0.00038, # regularisation of sparsity of the final matrix
        "lambda_kernal" : 0.8,
        "lr":0.0022,
        "optim_type":'AdamW',
        "drop": 0.7,
        "dot_scale": 0.4}

    conf_Douban={
        "dataset": "Douban", # datset name; options( "ML-1M", "ML-100K", "Douban")
        "n_hid" : 500, #500 # size of hidden layers
        "n_dep" : 5, # depth on hidden layers
        "n_dim" : 5, # inner AE embedding size
        "k_len" : 3, # length of kernal in NeighbourLayer 
        "n_layers" : 3, # number of hidden layers

        # Hyperparameters to tune for specific case
        "max_epoch_p" : 100, # max number of epochs for pretraining
        "patience_p" : 10, #5 # number of consecutive rounds of early stopping condition before actual stop for pretraining
        "tol_p" : 1e-4, # minimum threshold for the difference between consecutive values of train rmse, used for early stopping, for pretraining
        "lambda_2" : 12., # regularisation of number or parameters
        "lambda_s" : 0.027, # regularisation of sparsity of the final matrix
        "lambda_kernal" : 0.1693,
        "lr":0.97,
        "optim_type":'LBFGS',
        "drop": 0.2045,
        "dot_scale": 0.1664}

    if dataset == 'ML-100K':
        return conf_ML_1M
    elif dataset == 'Douban':
        return conf_ML_100K
    else:
        return conf_Douban

validConfigFields=["dataset", "n_hid", "n_dep", "n_dim", "k_len", "n_layers", "max_epoch_p", "patience_p", "tol_p", "lambda_2", "lambda_s", "lambda_kernal", "lr", "drop"]
essentialFields=["dataset", "n_hid", "n_dep", "n_dim", "k_len", "n_layers", "max_epoch_p", "patience_p", "tol_p", "lambda_2", "lambda_s", "lambda_kernal"]
