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
from qhoptim.pyt import QHAdam
#You should set the path of each dataset!!!
data_root = "F:/Datasets/"
#dataset = "MICROSOFT"
dataset = "YAHOO"
#dataset = "YEAR"
#dataset = "CLICK"
#dataset = "HIGGS"


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



def GBDT_test(data,fold_n,num_rounds = 100000,bf=1,ff=1):
    model_type = "mort" if isMORT else "lgb"
    nFeatures = data.X_train.shape[1]
    early_stop = 100;    verbose_eval = 20
    
    #lr = 0.01;   
    bf = bf;    ff = ff

    if data.problem()=="classification":
        metric = 'auc'       #"rmse"
        params = {"objective": "binary", "metric": metric,'n_estimators': num_rounds,
        "bagging_fraction": bf, "feature_fraction": ff,'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1, 
              }
    else:
        metric = 'l2'       #"rmse"
        params = {"objective": "regression", "metric": metric,'n_estimators': num_rounds,
              "bagging_fraction": bf, "feature_fraction": ff, 'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1,
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
        if data.problem()=="classification":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=min(num_rounds//10,1000))
        pred_val = model.predict(data.X_test)
        #plot_importance(model)
        lgb.plot_importance(model, max_num_features=32)
        plt.title("Featurertances")
        plt.savefig(f"./results/{dataset}_feat_importance_.jpg")
        plt.show(block=False)
        #plt.close()

        fold_importance = pd.DataFrame()
        fold_importance["importance"] = model.feature_importances_
        fold_importance["feature"] = [i for i in range(nFeatures)]
        fold_importance["fold"] = fold_n
        #fold_importance.to_pickle(f"./results/{dataset}_feat_{fold_n}.pickle")
        print('best_score', model.best_score_)
        acc_train,acc_=model.best_score_['training'][metric], model.best_score_['valid_1'][metric]
    if data.X_test is not None:
        pred_val = model.predict(data.X_test)
        acc_ = ((data.y_test - pred_val) ** 2).mean()
        print(f'====== Best step: test={data.X_test.shape} ACCU@Test={acc_:.5f}')
    return acc_,fold_importance

def get_feature_info(data,fold_n):
    pkl_path = f"./results/{dataset}_feat_info_.pickle"
    nSamp,nFeat = data.X_train.shape[0],data.X_train.shape[1]
    if os.path.isfile(pkl_path):
        feat_info = pd.read_pickle(pkl_path)
    else:
        #fast GBDT to get feature importance
        nMostSamp,nMostFeat=100000.0,100.0
        bf = 1.0 if nSamp<=nMostSamp else nMostSamp/nSamp
        ff = 1.0 if nFeat<=nMostFeat else nMostFeat/nFeat
        accu,feat_info = GBDT_test(data,fold_n,num_rounds=1000,bf = bf,ff = ff)
        with open(pkl_path, "wb") as fp:
            pickle.dump(feat_info, fp)

    importance = torch.from_numpy(feat_info['importance'].values).float()
    fmax, fmin = torch.max(importance), torch.min(importance)
    weight = importance / fmax
    feat_info = data.OnFeatInfo(feat_info,weight)
    return feat_info

def cascade_LR():   #意义不大
    if config.cascade_LR:
        LinearRgressor = quantum_forest.Linear_Regressor({'cascade':"ridge"})
        y_New = LinearRgressor.BeforeFit((data.X_train, data.y_train),[(data.X_valid, data.y_valid),(data.X_test, data.y_test)])
        YY_train = y_New[0]
        YY_valid,YY_test = y_New[1],y_New[2]
    else:
        YY_train,YY_valid,YY_test = data.y_train, data.y_valid, data.y_test
    return YY_train,YY_valid,YY_test

def VisualAfterEpoch(epoch,visual,config,mse):
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


def NODE_test(data,fold_n,config,visual=None,feat_info=None):
    YY_train,YY_valid,YY_test = data.y_train, data.y_valid, data.y_test

    data.Y_mean,data.Y_std = YY_train.mean(), YY_train.std()
    #config.mean,config.std = mean,std
    print(f"======  NODE_test \ttrain={data.X_train.shape} valid={data.X_valid.shape} YY_train_mean={data.Y_mean:.3f} YY_train_std={data.Y_std:.3f}\n")
    in_features = data.X_train.shape[1]
    #config.tree_module = node_lib.ODST
    config.tree_module = quantum_forest.DeTree
    Learners,last_train_prediction=[],0
    qForest = quantum_forest.QForest_Net(in_features,config, feat_info=feat_info,visual=visual).to(config.device)   
    Learners.append(qForest)    

    if False:       # trigger data-aware init,作用不明显
        with torch.no_grad():
            res = qForest(torch.as_tensor(data.X_train[:1000], device=config.device))
    #if torch.cuda.device_count() > 1:        model = nn.DataParallel(model)

    #weight_decay的值需要反复适配       如取1.0e-6 还可以  0.61142-0.58948
    optimizer=QHAdam;           
    optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998),'lr':config.lr_base,'weight_decay':1.0e-8 }
    #一开始收敛快，后面要慢一些
    #optimizer = torch.optim.Adam;    optimizer_params = {'lr':config.lr_base }

    from IPython.display import clear_output
    loss_history, mse_history = [], []
    best_mse = float('inf')
    best_step_mse = 0
    early_stopping_rounds = 3000
    report_frequency = 1000
    config.eval_batch_size = 512 if config.leaf_output=="distri2CNN" else \
            512 if config.path_way=="TREE_map" else 1024

    wLearner=Learners[-1]
    trainer = quantum_forest.Experiment(
        config,data,
        model=wLearner, loss_function=F.mse_loss,
        experiment_name=config.experiment,
        warm_start=False,   
        Optimizer=optimizer,        optimizer_params=optimizer_params,
        verbose=True,      #True
        n_last_checkpoints=5
    )   
    
    trainer.SetLearner(wLearner)
    print(f"======  trainer.learner={trainer.model}\ntrainer.opt={trainer.opt}"\
        f"\n======  config={config.__dict__}")
    print(f"======  YY_train={np.linalg.norm(YY_train):.3f},mean={data.Y_mean:.3f} std={data.Y_std:.3f}")
    wLearner.AfterEpoch(isBetter=True, epoch=0)
    epoch,t0=0,time.time()
    for batch in node_lib.iterate_minibatches(data.X_train, YY_train, batch_size=config.batch_size,shuffle=True, epochs=float('inf')):
        metrics = trainer.train_on_batch(*batch, device=config.device)
        loss_history.append(metrics['loss'])
        if trainer.step%10==0:
            symbol = "^" if config.cascade_LR else ""
            print(f"\r============ {trainer.step}{symbol}\t{metrics['loss']:.5f}\tL1=[{wLearner.reg_L1:.4g}*{config.reg_L1}]"
            f"\tL2=[{wLearner.reg_L2:.4g}*{config.reg_Gate}]\ttime={time.time()-t0:.2f}\t"
            ,end="")
        if trainer.step % report_frequency == 0:
            epoch=epoch+1
            if torch.cuda.is_available():   torch.cuda.empty_cache()
            mse = trainer.AfterEpoch(epoch,data.X_valid,YY_valid,best_mse)            
            if mse < best_mse:
                best_mse = mse
                best_step_mse = trainer.step
                trainer.save_checkpoint(tag='best_mse')
            mse_history.append(mse)
            if config.average_training:
                trainer.load_checkpoint()  # last
                trainer.remove_old_temp_checkpoints()
            VisualAfterEpoch(epoch,visual,config,mse)
            if False and epoch%10==9: #有bug啊
                #YY_valid = YY_valid- prediction
                dict_info,train_pred = trainer.evaluate_mse(data.X_train, YY_train, device=config.device, batch_size=config.eval_batch_size)
                #last_train_prediction = last_train_prediction+train_pred
                mse_train = dict_info["mse"]                
                YY_train = YY_train-train_pred
                mean,std = YY_train.mean(), YY_train.std()
                qForest = quantum_forest.QForest_Net(in_features,config, feat_info=feat_info,visual=visual).to(config.device)   
                #Learners.append(qForest)
                wLearner=qForest#Learners[-1]
                print(f"NODE_test::Expand@{epoch} eval_train={mse_train:.2f} YY_train={np.linalg.norm(YY_train)}")
                trainer.SetModel(wLearner)

        if trainer.step>50000:
            break
        if trainer.step > best_step_mse + early_stopping_rounds:
            print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
            print("Best step: ", best_step_mse)
            print(f"Best Val MSE: {best_mse:.5f}")
            break
    if data.X_test is not None:
        mse = trainer.AfterEpoch(epoch,data.X_test, YY_test,best_mse,isTest=True) 
        if False:
            if torch.cuda.is_available():  torch.cuda.empty_cache()
            trainer.load_checkpoint(tag='best_mse')
            t0=time.time()
            dict_info,prediction = trainer.evaluate_mse(data.X_test, YY_test, device=config.device, batch_size=config.eval_batch_size)
            if config.cascade_LR:
                prediction=LinearRgressor.AfterPredict(data.X_test,prediction)
            #prediction = prediction*data.accu_scale+data.Y_mu_0
            prediction = data.Y_trans(prediction)
            mse = ((data.y_test - prediction) ** 2).mean()
            #mse = dict_info["mse"]
            reg_Gate = dict_info["reg_Gate"]
            print(f'====== Best step: {trainer.step} test={data.X_test.shape} ACCU@Test={mse:.5f} \treg_Gate:{reg_Gate:.4g}time={time.time()-t0:.2f}' )
        best_mse = mse
    return best_mse,mse



def Fold_learning(fold_n,data,config,visual):
    t0 = time.time()
    if config.model=="QForest":
        if config.feat_info == "importance":
            feat_info = get_feature_info(data,fold_n)            
        else:
            feat_info = None
        accu,_ = NODE_test(data,fold_n,config,visual,feat_info)
    elif config.model=="GBDT":
        accu,_ = GBDT_test(data,fold_n)
    else:        #"LinearRegressor"    
        model = quantum_forest.Linear_Regressor({'cascade':"ridge"})
        accu,_ = model.fit((data.X_train, data.y_train),[(data.X_test, data.y_test)])

    print(f"\n======\n====== Fold_{fold_n}\tACCURACY={accu:.5f},time={time.time() - t0:.2f} ====== \n======\n")
    return

if __name__ == "__main__":
    data = quantum_forest.TabularDataset(dataset,data_path=data_root, random_state=1337, quantile_transform=True, quantile_noise=1e-3)
    #data = quantum_forest.TabularDataset(dataset,data_path=data_root, random_state=1337, quantile_transform=True)
    config = quantum_forest.QForest_config(data,0.002,feat_info="importance")   #,feat_info="importance"
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)

    config.model="QForest"      #"QForest"            "GBDT" "LinearRegressor"    
    if dataset=="YAHOO" or dataset=="MICROSOFT" or dataset=="CLICK" or dataset=="HIGGS":
        config,visual = InitExperiment(config, 0)
        data.onFold(0,config,pkl_path=f"{data_root}{dataset}/FOLD_Quantile_.pickle")
        Fold_learning(0,data, config,visual)
    else:
        nFold = 5 if dataset != "HIGGS" else 20
        folds = KFold(n_splits=nFold, shuffle=True)
        index_sets=[]
        for fold_n, (train_index, valid_index) in enumerate(folds.split(data.X)):
            index_sets.append(valid_index)
        for fold_n in range(len(index_sets)):
            config, visual = InitExperiment(config, fold_n)
            train_list=[]
            for i in range(nFold):
                if i==fold_n:           #test
                    continue
                elif i==fold_n+1:       #valid                
                    valid_index=index_sets[i]
                else:
                    train_list.append(index_sets[i])
            train_index=np.concatenate(train_list)
            print(f"train={len(train_index)} valid={len(valid_index)} test={len(index_sets[fold_n])}")

            data.onFold(fold_n,config,train_index=train_index, valid_index=valid_index,test_index=index_sets[fold_n],pkl_path=f"{data_root}{dataset}/FOLD_{fold_n}.pickle")
            Fold_learning(fold_n,data,config,visual)
            break
            



