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
import torch, torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import KFold
from qhoptim.pyt import QHAdam
from GBDT_test import *
#You should set the path of each dataset!!!
# data_root = "F:/Datasets/"


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
        visual.UpdateLoss(title=f"Accuracy on \"{config.dataset}\"",legend=f"{config.experiment}", loss=mse,yLabel="Accuracy")


def QF_test(data,fold_n,config,visual=None,feat_info=None):
    YY_train,YY_valid,YY_test = data.y_train, data.y_valid, data.y_test

    data.Y_mean,data.Y_std = YY_train.mean(), YY_train.std()
    #config.mean,config.std = mean,std
    print(f"======  QF_test \ttrain={data.X_train.shape} valid={data.X_valid.shape} YY_train_mean={data.Y_mean:.3f} YY_train_std={data.Y_std:.3f}\n")
    in_features = data.X_train.shape[1]
    config.in_features = in_features
    #config.tree_module = ODST
    config.tree_module = quantum_forest.DeTree
    if config.QF_fit>0:    #sklearn-like style
        learner = quantum_forest.QuantumForest(config,data,feat_info=feat_info,visual=visual).fit(data.X_train, YY_train, eval_set=[(data.X_valid,YY_valid)])
        trainer,best_mse = learner.trainer,learner.best_score
        epoch = config.nMostEpochs
    else:
        Learners,last_train_prediction=[],0
        qForest = quantum_forest.QF_Net(in_features,config, feat_info=feat_info,visual=visual).to(config.device)   
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
        #report_frequency = 10
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
        config.trainer = trainer
        
        trainer.SetLearner(wLearner)
        print(f"======  trainer.learner={trainer.model}\ntrainer.opt={trainer.opt}"\
            f"\n======  config={config.__dict__}")
        print(f"======  X_train={data.X_train.shape},YY_train={YY_train.shape}")
        print(f"======  |YY_train|={np.linalg.norm(YY_train):.3f},mean={data.Y_mean:.3f} std={data.Y_std:.3f}")
        wLearner.AfterEpoch(isBetter=True, epoch=0)
        epoch,t0=0,time.time()
        for batch in quantum_forest.experiment.iterate_minibatch(data.X_train, YY_train, batch_size=config.batch_size,shuffle=True, epochs=float('inf')):
            metrics = trainer.train_on_batch(*batch, device=config.device)
            loss_history.append(metrics['loss'])
            if trainer.step%10==0:
                symbol = "^" if config.cascade_LR else ""
                print(f"\r============ {trainer.step}{symbol}\t{metrics['loss']:.5f}\tL1=[{wLearner.reg_L1:.4g}*{config.reg_L1}]"
                f"\tL2=[{wLearner.L_gate:.4g}*{config.reg_Gate}]\ttime={time.time()-t0:.2f}\t"
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
                    qForest = quantum_forest.QF_Net(in_features,config, feat_info=feat_info,visual=visual).to(config.device)   
                    #Learners.append(qForest)
                    wLearner=qForest#Learners[-1]
                    print(f"QF_test::Expand@{epoch} eval_train={mse_train:.2f} YY_train={np.linalg.norm(YY_train)}")
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
    trainer.save_checkpoint(tag=f'last_{mse:.6f}')
    return best_mse,mse

def Fold_learning(fold_n,data,config,visual):
    t0 = time.time()
    if config.model=="QForest":
        if config.feat_info == "importance":
            feat_info = get_feature_info(data,fold_n)            
        else:
            feat_info = None
        accu,_ = QF_test(data,fold_n,config,visual,feat_info)
    elif config.model=="GBDT" or config.model=="Catboost" or config.model=="XGBoost" or config.model=="LightGBM":
        accu,_ = GBDT_test(config,data,fold_n,num_rounds=config.nMostEpochs)
    else:        #"LinearRegressor"    
        model = quantum_forest.Linear_Regressor({'cascade':"ridge"})
        accu,_ = model.fit((data.X_train, data.y_train),[(data.X_test, data.y_test)])

    print(f"\n======\n====== Fold_{fold_n}@{data.name}\t{data.problem()}"\
            f"\tACCURACY={accu:.5f},time={time.time() - t0:.2f} ====== \n======\n")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--data_root',required=True)
    parser.add_argument('--dataset', default='CLICK',help="MICROSOFT,YAHOO,YEAR,CLICK,HIGGS,EPSILON")
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--model', default="QForest", help='QForest,GBDT,LinearRegressor')
    parser.add_argument('--learning_rate', default="0.001", type=float)
    parser.add_argument('--subsample', default="1", type=float)
    parser.add_argument('--QF_fit', default="1", type=int)
    parser.add_argument('--attention', default="eca_response", type=str)
    parser.add_argument('--scale', default="medium",help='small，medium，large', type=str)
    args = parser.parse_args()
    print(f"===== {args.__dict__}")
    dataset = args.dataset
    data = quantum_forest.TabularDataset(dataset,data_path=args.data_root, random_state=1337, quantile_transform=True, quantile_noise=1e-3)
    #data = quantum_forest.TabularDataset(dataset,data_path=data_root, random_state=1337, quantile_transform=True)
    
    config = quantum_forest.QForest_config(data,0.002)   #,feat_info="importance","attention"
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)
    config.model=args.model      #"QForest"            "GBDT" "LinearRegressor"    
    if config.model[0]=="Q":    config.model="QForest"
    config.lr_base = args.learning_rate
    config.dataset = args.dataset
    config.bagging_fraction = args.subsample
    config.nMostEpochs = args.iterations
    config.QF_fit = args.QF_fit
    config.attention_alg = args.attention
    if args.scale == "small":
        config.depth, config.batch_size, config.nTree = 4, 256, 256
    elif args.scale == "medium":
        config.depth, config.batch_size, config.nTree = 5, 512, 1024
    elif args.scale == "large":
        config.depth, config.batch_size, config.nTree = 5, 512, 2048

    if dataset=="YAHOO" or dataset=="MICROSOFT" or dataset=="CLICK" or dataset=="HIGGS" or dataset=="EPSILON":
        config,visual = quantum_forest.InitExperiment(config, 0)
        data.onFold(0,config,pkl_path=f"{args.data_root}{dataset}/FOLD_Quantile_{config.model}.pickle")
        Fold_learning(0,data, config,visual)
    else:   #"YEAR"
        nFold = 5 if dataset != "HIGGS" else 20
        folds = KFold(n_splits=nFold, shuffle=True)
        index_sets=[]
        for fold_n, (train_index, valid_index) in enumerate(folds.split(data.X)):
            index_sets.append(valid_index)
        for fold_n in range(len(index_sets)):
            config, visual = quantum_forest.InitExperiment(config, fold_n)
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

            data.onFold(fold_n,config,train_index=train_index, valid_index=valid_index,test_index=index_sets[fold_n],pkl_path=f"{args.data_root}{dataset}/FOLD_{fold_n}_{config.model}.pickle")
            Fold_learning(fold_n,data,config,visual)
            break
            



