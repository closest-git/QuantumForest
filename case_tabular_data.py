'''

'''

import os, sys
import time
sys.path.insert(0, './python-package/')
isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
if isMORT:
    sys.path.insert(1, 'E:/LiteMORT/python-package/')
    import litemort
    from litemort import *
    print(f"litemort={litemort.__version__}")
import numpy as np
import matplotlib.pyplot as plt
import node_lib
import quantum_forest
import pandas as pd
import pickle
import torch, torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.model_selection import KFold

#You should set the path of each dataset!!!
data_root = "F:/Datasets/"
#dataset = "MICROSOFT"
dataset = "YAHOO"
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def InitExperiment(config,fold_n):
    config.experiment = f'{config.data_set}_{config.model_info()}_{fold_n}'   #'year_node_shallow'
    #experiment = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}_{:0>2d}'.format(experiment, *time.gmtime()[:5])
    #visual = quantum_forest.Visdom_Visualizer(env_title=config.experiment)
    visual = quantum_forest.Visualize(env_title=config.experiment)
    visual.img_dir = "./results/images/"
    print("experiment:", config.experiment)
    log_path=f"logs/{config.experiment}"
    if os.path.exists(log_path):        #so strange!!!
        import shutil
        print(f'experiment {config.experiment} already exists, DELETE it!!!')
        shutil.rmtree(log_path)
    return config,visual


def LoadData( ):
    pkl_path = f'./data/{dataset}.pickle'
    if False and os.path.isfile(pkl_path):
        print("====== LoadData@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data = pickle.load(fp)
    else:
        data = quantum_forest.Dataset(dataset, random_state=1337, quantile_transform=True, quantile_noise=1e-3)
        #data = node_lib.Dataset("HIGGS",data_path="F:/Datasets/",random_state=1337, quantile_transform=True, quantile_noise=1e-3)
        in_features = data.X_train.shape[1]
        mu, std = data.y_train.mean(), data.y_train.std()
        normalize = lambda x: ((x - mu) / std).astype(np.float32)
        data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])
        print("mean = %.5f, std = %.5f" % (mu, std))
        with open(pkl_path, "wb") as fp:
            pickle.dump(data,fp)
    return data

def plot_importance(model,nMax=30):
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(model, max_num_features=nMax)
    plt.title("Featurertances")
    plt.show()

def GBDT_test(data,fold_n):
    model_type = "mort" if isMORT else "lgb"
    nFeatures = data.X_train.shape[1]
    some_rows = 10000
    early_stop = 100;    verbose_eval = 20
    metric = 'l2'       #"rmse"
    num_rounds = 100000; nLeaf = 32
    lr = 0.02;    bf = 0.51;    ff = 0.81

    params = {"objective": "regression", "metric": metric,
              "num_leaves": nLeaf, "learning_rate": lr, 'n_estimators': num_rounds,
              "bagging_freq": 1, "bagging_fraction": bf, "feature_fraction": ff, 'min_data_in_leaf': 10000,
              'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1, "elitism": 0
              }
    print(f"====== GBDT_test\tparams={params}")
    X_train, y_train = data.X_train, data.y_train
    X_valid, y_valid = data.X_valid, data.y_valid
    X_test, y_test = data.X_test, data.y_test
    if not np.isfortran(X_train):   #Very important!!! mort need COLUMN-MAJOR format
        X_train = np.asfortranarray(X_train)
        X_valid = np.asfortranarray(X_valid)
    #X_train, X_valid = pd.DataFrame(X_train), pd.DataFrame(X_valid)
    print(f"GBDT_test\ttrain={X_train.shape} valid={X_valid.shape}")
    #print(f"X_train=\n{X_train.head()}\n{X_train.tail()}")
    if model_type == 'mort':
        params['verbose'] = 667
        model = LiteMORT(params).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        #y_pred_valid = model.predict(X_valid)
        #y_pred = model.predict(X_test)

    if model_type == 'lgb':
        #params['verbose'] = 0   #667
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=1000)
        pred_val = model.predict(data.X_test)
        plot_importance(model)
        fold_importance = pd.DataFrame()
        fold_importance["importance"] = model.feature_importances_
        fold_importance["feature"] = [i for i in range(nFeatures)]
        fold_importance["fold"] = fold_n
        fold_importance.to_pickle(f"./results/{dataset}_feat_{fold_n}.pickle")
        print('best_score', model.best_score_)
        acc_train,acc_=model.best_score_['training'][metric], model.best_score_['valid_1'][metric]
    if data.X_test is not None:
        pred_val = model.predict(data.X_test)
        acc_ = ((data.y_test - pred_val) ** 2).mean()
        print(f'====== Best step: test={data.X_test.shape} ACCU@Test={acc_:.5f}')
    return acc_,acc_train

def dump_model_params(model):
    nzParams = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nzParams += param.nelement()
            print(f"\t{name}={param.nelement()}")
    print(f"========All parameters={nzParams}")
    return nzParams


def NODE_test(data,fold_n,config,visual=None,feat_info=None):
    #print(f"======  NODE_test depth={depth},batch={batch_size},nTree={nTree}\n")
    print(f"======  NODE_test \ttrain={data.X_train.shape} valid={data.X_valid.shape} \n======  config={config}\n")
    in_features = data.X_train.shape[1]
    #config.tree_module = node_lib.ODST
    config.tree_module = quantum_forest.DeTree
    if True:#,feat_info=feat_info
        model = quantum_forest.QForest_Net(in_features,config, feat_info=feat_info).to(device)       #
    else:
        model = nn.Sequential(
            DecisionBlock(in_features, config, flatten_output=False,feat_info=feat_info),
            #MultiBlock(in_features, config, flatten_output=False,feat_info=feat_info),
            #node_lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
        ).to(device)
    model.visual = visual
    print(model)
    dump_model_params(model)

    if False:       # trigger data-aware init,作用不明显
        with torch.no_grad():
            res = model(torch.as_tensor(data.X_train[:1000], device=device))

    #if torch.cuda.device_count() > 1:        model = nn.DataParallel(model)

    from qhoptim.pyt import QHAdam
    #weight_decay的值需要反复适配       如取1.0e-6 还可以  0.61142-0.58948
    optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998),'lr':0.002 }
    trainer = quantum_forest.Trainer(
        model=model, loss_function=F.mse_loss,
        experiment_name=config.experiment,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        verbose=False,      #True
        n_last_checkpoints=5
    )
    from IPython.display import clear_output
    loss_history, mse_history = [], []
    best_mse = float('inf')
    best_step_mse = 0
    early_stopping_rounds = 3000
    report_frequency = 1000
    eval_batch_size = 512 if config.path_way=="TREE_map" else 1024

    print(f"trainer.model={trainer.model}\ntrainer.opt={trainer.opt}")
    model.AfterEpoch(isBetter=True, epoch=0)
    epoch,t0=0,time.time()
    for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, batch_size=config.batch_size,
                                         shuffle=True, epochs=float('inf')):

        metrics = trainer.train_on_batch(*batch, device=device)
        loss_history.append(metrics['loss'])
        if trainer.step%10==0:
            print(f"\r============ {trainer.step}\t{metrics['loss']:.5f}\tL1=[{model.reg_L1:.4g}*{config.reg_L1}]"
            f"\tL2=[{model.reg_L2:.4g}*{config.reg_Gate}]\ttime={time.time()-t0:.2f}\t"
            ,end="")
        if trainer.step % report_frequency == 0:
            epoch=epoch+1
            if torch.cuda.is_available():   torch.cuda.empty_cache()
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')
            dict_info = trainer.evaluate_mse(data.X_valid, data.y_valid, device=device, batch_size=eval_batch_size)
            mse = dict_info["mse"]

            model.AfterEpoch(isBetter=mse < best_mse, accu=mse,epoch=epoch)
            if mse < best_mse:
                best_mse = mse
                best_step_mse = trainer.step
                trainer.save_checkpoint(tag='best_mse')
            mse_history.append(mse)

            trainer.load_checkpoint()  # last
            trainer.remove_old_temp_checkpoints()

            if visual is None:
                if config.plot_train:
                    clear_output(True)
                    plt.figure(figsize=[18, 6])
                    plt.subplot(1, 2, 1)
                    plt.plot(loss_history)
                    plt.title('Loss')
                    plt.grid()
                    plt.subplot(1, 2, 2)
                    plt.plot(mse_history)
                    plt.title('MSE')
                    plt.grid()
                    plt.show()
            else:
                visual.UpdateLoss(title=f"Accuracy on \"{dataset}\"",legend=f"{config.experiment}", loss=mse,yLabel="Accuracy")
            reg_Gate = dict_info["reg_Gate"] 
            print(f"\nloss_{trainer.step}\t{metrics['loss']:.5f}\treg_Gate:{reg_Gate:.4g}\tVal MSE:{mse:.5f}" )   
                     
        if trainer.step>50000:
            break
        if trainer.step > best_step_mse + early_stopping_rounds:
            print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
            print("Best step: ", best_step_mse)
            print("Best Val MSE: %0.5f" % (best_mse))
            break
    if data.X_test is not None:
        if torch.cuda.is_available():  torch.cuda.empty_cache()
        trainer.load_checkpoint(tag='best_mse')
        t0=time.time()
        dict_info = trainer.evaluate_mse(data.X_test, data.y_test, device=device, batch_size=eval_batch_size)
        mse = dict_info["mse"]
        reg_Gate = dict_info["reg_Gate"]
        print(f'====== Best step: {trainer.step} test={data.X_test.shape} ACCU@Test={mse:.5f} \treg_Gate:{reg_Gate:.4g}time={time.time()-t0:.2f}' )
        best_mse = mse
    return best_mse,mse


def Fold_learning(fold_n,data,config,visual):
    t0 = time.time()
    if config.model=="QForest":
        if config.feat_info == "importance":
            feat_info = pd.read_pickle(f"./results/{dataset}_feat_.pickle")
            importance = torch.from_numpy(feat_info['importance'].values).float()
            fmax, fmin = torch.max(importance), torch.min(importance)
            weight = importance / fmax
            feat_info = data.OnFeatInfo(feat_info,weight)
        else:
            feat_info = None
        accu,_ = NODE_test(data,fold_n,config,visual,feat_info)
    else:
        accu,_ = GBDT_test(data,fold_n)
    print(f"\n======\n====== Fold_{fold_n}\tACCURACY={accu:.5f},time={time.time() - t0:.2f} ====== \n======\n")
    return

if __name__ == "__main__":
    #data = LoadData()
    data = quantum_forest.TabularDataset(dataset,data_path=data_root, random_state=1337, quantile_transform=True, quantile_noise=1e-3)
    config = quantum_forest.QForest_config(dataset,0.002,feat_info="importance")   #,feat_info="importance"
    random_state = 42
    quantum_forest.OnInitInstance(random_state)

    config.model="QForest"      #"QForest"            "GBDT"
    if dataset=="YAHOO" or dataset=="MICROSOFT":
        config,visual = InitExperiment(config, 0)
        data.onFold(0,config,pkl_path=f"{data_root}{dataset}/FOLD_Quantile_.pickle")
        Fold_learning(0,data, config,visual)
    else:
        nFold = 5
        folds = KFold(n_splits=nFold, shuffle=True)
        index_sets=[]
        for fold_n, (train_index, valid_index) in enumerate(folds.split(data.X)):
            index_sets.append(valid_index)
        for fold_n in range(len(index_sets)):
            config, visual = InitExperiment(config, fold_n)
            valid_index=index_sets[(fold_n+1)%nFold]
            train_index=np.concatenate([index_sets[(fold_n+2)%nFold],index_sets[(fold_n+3)%nFold],index_sets[(fold_n+4)%nFold]])
            print(f"train={len(train_index)} valid={len(valid_index)} test={len(index_sets[fold_n])}")

            data.onFold(fold_n,train_index=train_index, valid_index=valid_index,test_index=index_sets[fold_n],pkl_path=f"{data_root}{dataset}/FOLD_{fold_n}.pickle")
            Fold_learning(fold_n,data,config,visual)



