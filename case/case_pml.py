'''
@Author: Yingshi Chen

@Date: 2020-03-15 19:28:48
@
# Description: 
'''

import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, './')
from main_tabular_data import *
import argparse 
import glob
from io import StringIO
#import pandas_profiling

class PML_dataset(quantum_forest.TabularDataset):
    def file2key(self,file):
        name = os.path.basename(file)
        infos = name.split("_")
        #Ey ex(分量)    it1(坐标)   right(上，下，左，右)   Groupth1_a0_59_PMLnum40
        #print(name)
        key=(infos[1],infos[2],infos[3])
        isY = True if "ot" in key else False
        return isY,key,name

    def problem(self):
        return "regression"
        #return "regression_N"

    def load_files(self,isGroup=True):
        self.isTrain = True
        self.isSample = False
        listX_,listY_=[],[]
        nPt=0
        if isGroup:
            group_files=[]
            for group_path in self.data_path:
                points,files = {}, [x[0] for x in os.walk(group_path) if x[0]!=group_path]   #glob.glob(f"{self.data_path}*")  
                group_files.extend(files)              
        else:
            group_files = self.data_path
        nPoint = len(group_files)
        for pt_files in group_files:
            #if pt_files==group_path:                continue
            X_,Y_,_ = self.load_files_v0(pt_files+"/")
            if self.isPrecit:
                assert X_.shape==(1500,12) 
                Y_ = None
            else:
                assert X_.shape==(1500,12) and Y_.shape==(1500,3)
            listX_.append(X_);      listY_.append(Y_)
            nPt = nPt+1
            if nPt>=self.nMostPt:   
                break
        if len(listX_)>0:            
            self.X = np.vstack(listX_)
            if not self.isPrecit:
                self.Y = np.vstack(listY_)
                self.Y=self.Y[:,self.ex_ey_hz]                
            else:
                self.Y = np.zeros((self.X.shape[0],3))
            self.nFeature = self.X.shape[1]
            num_features = self.X.shape[1]
            #self.X=self.X*1.0e6;            self.Y=self.Y*1.0e6
        else:
            print("Failed to load data@{self.data_path}!!!")
        lenY = np.linalg.norm(self.Y)
        if False:   #时间太长，pandas_profiling低效
            df = pd.DataFrame(self.X)
            print(df.head())
            pfr = pandas_profiling.ProfileReport(df)
            pfr.to_file("./example.html")

        print(f"X_={self.X.shape} {self.X[:5,:]}\n{self.X[-5:,:]}")
        print(f"Y_={self.Y.shape} |Y|={lenY:.4f}\t{self.Y[:50]}\n{self.Y[-50:]}")
        return 

    def load_files_v0(self,files_path,isMerge=True):
        points,list_of_files = {}, glob.glob(f"{files_path}*")
        nFile = len(list_of_files)
        assert nFile==15
        X_,Y_ = pd.DataFrame(),pd.DataFrame()
        for id,file in enumerate(list_of_files) :
            assert os.path.isfile(file)
            isY,key,name = self.file2key(file)
            #if not "right" in key:
            #    continue
            with open(file, 'r') as file :
                filedata = file.read()           
            #filedata = StringIO(filedata.replace('E', 'e'))     #真麻烦
            filedata = StringIO(filedata.replace('D', 'e'))     #真麻烦
            #print(filedata)
            df = pd.read_csv(filedata,header = None,dtype=np.float)    
            columns = df.columns
            #df.rename(columns={columns[0]:key},inplace=True) 
            #print(f"{name}={df.shape}\n{df}")       
            if isY:
                Y_[key]=df[0]
            else:
                X_[key]=df[0]

        num_features = X_.shape[1]
        assert X_.shape[0]==Y_.shape[0]
        if isMerge:
            return X_.values.astype('float32'), Y_.values.astype('float32'),None
        else:
            print(f"X_={X_.shape} {X_.head()}\n{X_.tail()}")
            print(f"Y_={Y_.shape} {Y_.head()}\n{Y_.tail()}")            
            self.X, self.Y = X_.values.astype('float32'), Y_.values.astype('float32')
            self.Y=self.Y[:,2]
            self.nFeature = self.X.shape[1]
        #self.nClasses = 0
        return 

    def __init__(self, dataset, data_path, normalize=False,nMostPt=100000,isPrecit=False,ex_ey_hz=0,
                 quantile_transform=False, output_distribution='normal', quantile_noise=1.0e-3, **kwargs):
        self.random_state = 42
        self.quantile_noise = quantile_noise
        self.name = f"{dataset}"
        self.data_path = data_path
        self.nMostPt = nMostPt
        self.isPrecit = isPrecit
        self.ex_ey_hz=ex_ey_hz         #output component

        self.load_files()
        self.zero_feats=[]        
        

def predict_(args):
    model = args.model
    if not os.path.isfile(args.model):
        print(f"'{args.model}' is not a valid MODEL file!!!")
        return
    data_paths = [args.predict]
    data = PML_dataset("PML_predict",data_path=data_paths,nMostPt=1,isPrecit=True)
    config = quantum_forest.QForest_config(data,0.002,feat_info="importance")   #,feat_info="importance"
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)
    config.err_relative = True
    config.model="QForest"      #"QForest"            "GBDT" "LinearRegressor"   
    config.tree_module = quantum_forest.DeTree
    config.task = "predict"
    config, visual = InitExperiment(config, 0)
    config.response_dim = 3
    config.feat_info = None
    config.eval_batch_size = 512

    qForest = quantum_forest.QForest_Net(data.nFeature,config).to(config.device)   
    trainer = quantum_forest.Experiment(
        config,data,model=qForest, loss_function=F.mse_loss,verbose=True
    )
    trainer.load_checkpoint(tag=None,path=args.model)  
    epoch,best_mse = -1,-1#trainer.step
    mse = trainer.AfterEpoch(epoch,data.X,data.Y,best_mse,isPredict=True)  
    sample_dir = args.predict
    pass

if __name__ == "__main__":
    dataset = "PML"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to pretrained model', default='E:/xiada/FengNX/pml_trained.pth')
    parser.add_argument('--predict', type=str, help='the directory include testing sample files', 
            default="E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型上边的全部离散P点/")
    args, unknown = parser.parse_known_args()
      
    if False and hasattr(args,'predict') and os.path.isdir(args.predict):
        predict_(args)
        exit(-1)

    data_paths = [
            #"E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型上边的全部离散P点/",
            #"E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型右边的全部离散P点/",
            "F:/PML_datas/2028组/模型上边的全部离散P点/",
            #"E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型下边的全部离散P点/",
            #"E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型左边的全部离散P点/",
        ]
    data = PML_dataset(dataset,data_path=data_paths,ex_ey_hz=2)        #,nMostPt=10
    config = quantum_forest.QForest_config(data,0.002,feat_info="importance")   #,feat_info="importance"    
    data_root = "F:/Datasets/"   #args.data_root
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)
    config.err_relative = True
    config.model="QForest"      #"QForest"            "GBDT" "LinearRegressor"  
    #config.path_way = "OBLIVIOUS_map"
    #config.batch_size = 256
    #config.nTree = 512
    
    nFold = 5 if dataset != "HIGGS" else 20
    folds = KFold(n_splits=nFold, shuffle=True)
    index_sets=[]
    for fold_n, (train_index, valid_index) in enumerate(folds.split(data.X)):
        index_sets.append(valid_index)
    for fold_n in range(len(index_sets)):
        config, visual =  quantum_forest.InitExperiment(config, fold_n)
        config.response_dim = 3
        config.feat_info = None
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
        data.Y_mean,data.Y_std = data.y_train.mean(), data.y_train.std()
        if False:
            Fold_learning(fold_n,data,config,visual)
        else:     
            config.in_features = data.X_train.shape[1]
            config.tree_module = quantum_forest.DeTree       
            learner = quantum_forest.QuantumForest(config,data,feat_info=None,visual=visual).   \
                fit(data.X_train, data.y_train, eval_set=[(data.X_valid,data.y_valid)])
            best_score = learner.best_score,0
        break
