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
#import node_lib
import quantum_forest
import pandas as pd
import pickle
import argparse
import lightgbm as lgb
import catboost as cat
import xgboost as xgb
from sklearn.model_selection import KFold

def Catboost_train(config,data,param_0,fold_n):
    metric = param_0['metric']
    num_rounds = param_0['n_estimators']
    nFeatures = data.X_train.shape[1]
    X_train, y_train = data.X_train, data.y_train
    X_valid, y_valid = data.X_valid, data.y_valid
    X_test, y_test = data.X_test, data.y_test
    params = {
        #'devices': [0],
        'logging_level': 'Info',
        #'use_best_model': False,
        #'bootstrap_type': 'Bernoulli',
        'random_seed': 42,
        'n_estimators': num_rounds,
    }
    params['custom_metric'] = 'Accuracy'
    if data.problem()=="classification":
        if data.nClasses==2:
            params['loss_function'] = 'Logloss'
        else:
            params['loss_function'] = 'MultiClass'
        model = cat.CatBoostClassifier(**params)
    else:
        params['loss_function'] = 'RMSE'    
        model = cat.CatBoostRegressor(iterations=num_rounds,loss_function='RMSE')
    #model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=min(num_rounds//10,100))
    train_pool = cat.Pool(X_train, y_train) 
    valid_pool  = cat.Pool(X_valid,y_valid) 
    model.fit(train_pool, eval_set=valid_pool)
    #pred_val = model.predict(data.X_test)
    
    return model,None

def XGBoost_train(config,data,params,fold_n):
    metric = params['metric']
    num_rounds = params['n_estimators']
    nFeatures = data.X_train.shape[1]
    X_train, y_train = data.X_train, data.y_train
    X_valid, y_valid = data.X_valid, data.y_valid
    X_test, y_test = data.X_test, data.y_test
    if data.problem()=="classification":
        if data.nClasses==2:
            params["objective"] = "binary:logistic"
            params['eval_metric'] = 'error'
        else:
            params["objective"] = "multi:softmax"
            params['eval_metric'] = 'merror'
        model = xgb.XGBClassifier(**params)
    else:
        params["objective"] = "reg:linear"
        params['eval_metric'] = 'error'
        model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=min(num_rounds//10,100))
    #model.fit(X_train, y_train)
    #pred_val = model.predict(data.X_test)
    
    return model,None

def lgb_train(config,data,params,fold_n):
    metric = params['metric']
    num_rounds = params['n_estimators']
    nFeatures = data.X_train.shape[1]
    X_train, y_train = data.X_train, data.y_train
    X_valid, y_valid = data.X_valid, data.y_valid
    X_test, y_test = data.X_test, data.y_test
    if data.problem()=="classification":
        model = lgb.LGBMClassifier(**params)
    else:
        model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=min(num_rounds//10,100))
    pred_val = model.predict(data.X_test)
    #plot_importance(model)
    lgb.plot_importance(model, max_num_features=32)
    plt.title("Featurertances")
    plt.savefig(f"./results/{config.dataset}_feat_importance_.jpg")
    #plt.show(block=False)
    plt.close()

    fold_importance = pd.DataFrame()
    fold_importance["importance"] = model.feature_importances_
    fold_importance["feature"] = [i for i in range(nFeatures)]
    fold_importance["fold"] = fold_n
    #fold_importance.to_pickle(f"./results/{config.dataset}_feat_{fold_n}.pickle")
    print('best_score', model.best_score_)
    acc_train,acc_=model.best_score_['training'][metric], model.best_score_['valid_1'][metric]
    return model,fold_importance

def GBDT_test(config,data,fold_n,num_rounds = 100000):
    model_type = "mort" if isMORT else config.model
    nFeatures = data.X_train.shape[1]
    early_stop = 100;    verbose_eval = 20
    
    lr = config.lr_base;    #default=0.1
    bf = config.bagging_fraction;    ff = config.feature_fraction   #default=1.0,1.0

    if data.problem()=="classification":
        metric = 'auc'       #"rmse"
        params = {"objective": "binary", "metric": metric,'n_estimators': num_rounds,"bagging_freq":1,'learning_rate':lr,
        "bagging_fraction": bf, "feature_fraction": ff,'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1, 
              }
    else:
        metric = 'l2'       #"rmse"
        params = {"objective": "regression", "metric": metric,'n_estimators': num_rounds,"bagging_freq":1,'learning_rate':lr,
              "bagging_fraction": bf, "feature_fraction": ff, 'verbose_eval': verbose_eval, "early_stopping_rounds": early_stop, 'n_jobs': -1,
              }
    print(f"====== GBDT_test\tparams={params}\n")
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
    elif model_type == 'XGBoost':
        model,fold_importance = XGBoost_train(config,data,params,fold_n)
    elif model_type == 'Catboost':
        model,fold_importance = Catboost_train(config,data,params,fold_n)
    else:    #if model_type == 'lgb':
        model,fold_importance = lgb_train(config,data,params,fold_n)
        # if data.problem()=="classification":
        #     model = lgb.LGBMClassifier(**params)
        # else:
        #     model = lgb.LGBMRegressor(**params)
        # model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=min(num_rounds//10,100))
        # pred_val = model.predict(data.X_test)
        # #plot_importance(model)
        # lgb.plot_importance(model, max_num_features=32)
        # plt.title("Featurertances")
        # plt.savefig(f"./results/{config.dataset}_feat_importance_.jpg")
        # #plt.show(block=False)
        # plt.close()

        # fold_importance = pd.DataFrame()
        # fold_importance["importance"] = model.feature_importances_
        # fold_importance["feature"] = [i for i in range(nFeatures)]
        # fold_importance["fold"] = fold_n
        # #fold_importance.to_pickle(f"./results/{config.dataset}_feat_{fold_n}.pickle")
        # print('best_score', model.best_score_)
        # acc_train,acc_=model.best_score_['training'][metric], model.best_score_['valid_1'][metric]
    if data.X_test is not None:
        pred_val = model.predict(data.X_test)
        if False:#config.err_relative:
            #nrm_Y = ((YY_) ** 2).mean()
            #mse = ((YY_ - prediction) ** 2).mean()/nrm_Y  
            lenY = np.linalg.norm(data.y_test) 
            acc_ = np.linalg.norm(data.y_test - pred_val)/lenY 
        else:
            acc_ = ((data.y_test - pred_val) ** 2).mean()
        print(f'====== Best step: test={data.X_test.shape} ACCU@Test={acc_:.5f}')
    return acc_,fold_importance

def get_feature_info(config,data,fold_n):
    pkl_path = f"./results/{config.dataset}_feat_info_.pickle"
    nSamp,nFeat = data.X_train.shape[0],data.X_train.shape[1]
    if os.path.isfile(pkl_path):
        feat_info = pd.read_pickle(pkl_path)
    else:
        #fast GBDT to get feature importance
        nMostSamp,nMostFeat=100000.0,100.0
        bf = 1.0 if nSamp<=nMostSamp else nMostSamp/nSamp
        ff = 1.0 if nFeat<=nMostFeat else nMostFeat/nFeat
        accu,feat_info = GBDT_test(data,fold_n,num_rounds=2000,bf = bf,ff = ff)
        with open(pkl_path, "wb") as fp:
            pickle.dump(feat_info, fp)

    importance = torch.from_numpy(feat_info['importance'].values).float()
    fmax, fmin = torch.max(importance), torch.min(importance)
    weight = importance / fmax
    feat_info = data.OnFeatInfo(feat_info,weight)
    return feat_info



            



