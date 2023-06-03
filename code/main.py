from imports import *
from intentRec import IntentRec
from dataReader import load_data
from loss import Loss
from utils import getConfig, getOptimizer, validConfigFields, essentialFields

def checkConfig(config_file):
    try:
        with open(config_file) as f:
            conf = json.load(f)
        confkeys=set(conf.keys())
        difflen=len(set(validConfigFields).difference(confkeys))
        if difflen!=0:
            if len(confkeys.intersection(essentialFields))==len(essentialFields):
                return conf
            else:
                raise Exception("Config has incorrect format/values")
        else:
            return conf
    except:
        raise Exception("invalid config file or file format")

def getParams():
    parser = ArgumentParser()
    parser.add_argument("-tp", "--train_file_path", dest="filepath")
    parser.add_argument("-cf", "--json_config_file", dest="config")
    args = parser.parse_args()
    args = vars(args)
    config =checkConfig(args['config'])
    filepath = args['filepath']
    return filepath, config

def setHyperParams(pars, h_params, n_m, n_u):
    for i in pars:
        if i in h_params:
            h_params[i]=pars[i]
    h_params['n_u']=n_u
    h_params['n_m']=n_m
    return h_params

def getData(path, dataset="ML-100K"):
     return load_data(path=path, dataset=dataset)

def getModel(params):
    model = IntentRec(n_u=params['n_u'], n_hid=params['n_hid'], n_dim=params['n_dim'], n_dep=params['n_dep'], k_len=params['k_len'],\
        n_layers=params['n_layers'], beta=params['beta'], lambda_s=params['lambda_s'], lambda_2=params['lambda_2'], \
        lambda_kernal=params['lambda_kernal'],lambda_kernal=params['lambda_reg'], lambda_kernal=params['lambda_reg2'],\
        beta=params['beta'], dropout=params['drop']).to(device)
    return model

def train(model, train_r, train_m, test_r, test_m, max_epoch_p=100, lr=0.001, tol_p=1e-4, patience_p=10, optim_type='AdamW', device=device, dot_scale=0.5):
    torch.cuda.empty_cache()
    time_cumulative = time()
    print('.-^-._'*4, 'Training Started', '.-^-._' * 5)
    print('Device: ', device)
    tic = time()
    best_rmse_ep, best_rmse = 0, float("inf")
    optimizer = getOptimizer(model.parameters(), optim_type=optim_type, lr=lr)
    print("Model: ", model)

    def closure():
        optimizer.zero_grad()
        x = torch.Tensor(train_r).double().to(device)
        m = torch.Tensor(train_m).double().to(device)
        model.train()
        pred, reg = model(x)
        model._updateProbs(pred, reg.item())
        loss = Loss().to(device)(pred, reg, m, x, dot_scale)
        loss.backward()
        return loss

    last_rmse = np.inf
    counter = 0
    print('.-^-._' * 12)
    try:
        for i in range(1, max_epoch_p+1):
            optimizer.step(closure)
            model.eval()
            t = time() - tic

            pre, _ = model(torch.Tensor(train_r).double().to(device))

            pre = pre.float().cpu().detach().numpy()

            error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
            test_rmse = np.sqrt(error)

            error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
            train_rmse = np.sqrt(error_train)
            
            if test_rmse<best_rmse:
                best_rmse = test_rmse
                best_rmse_ep = i

            if last_rmse-train_rmse < tol_p:
                counter += 1
            else:
                counter = 0

            last_rmse = train_rmse

            print('Epoch:', i, 'test rmse:', test_rmse, 'train rmse:', train_rmse)
            if patience_p == counter:
                break
            
        print('.-^-._' * 12)
        print('Time cumulative:', time()-time_cumulative, 'seconds')
        print('.-^-._'*4, 'Training Finished', '.-^-._' * 5)
        return best_rmse_ep, best_rmse
    
    except KeyboardInterrupt:
        print('Time cumulative:', time()-time_cumulative, 'seconds')
        print('.-^-._'*4, 'Training Finished', '.-^-._' * 5)
        return best_rmse_ep, best_rmse
        

if __name__=="__main__":
    filepath, args = getParams()
    h_params = getConfig(dataset=args['dataset'])
    n_m, n_u, train_r, train_m, test_r, test_m = getData(filepath, args['dataset'])
    h_params = setHyperParams(args, h_params, n_m, n_u)
    print("Config: ", h_params)
    model = getModel(h_params)
    best_rmse_ep, best_rmse = train(model=model, train_r=train_r, train_m=train_m, test_r=test_r, test_m=test_m, max_epoch_p=h_params['max_epoch_p'], lr=h_params["lr"], tol_p=h_params["tol_p"], optim_type=h_params['optim_type'], patience_p=h_params["patience_p"], dot_scale=h_params["dot_scale"])
    print('Best Epoch: ', best_rmse_ep, ' Best RMSE: ', best_rmse) 