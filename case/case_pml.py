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
import glob
from io import StringIO

class PML_dataset(quantum_forest.TabularDataset):
    def file2key(self,file):
        name = os.path.basename(file)
        infos = name.split("_")
        #Ey ex(分量)    it1(坐标)   right(上，下，左，右)   Groupth1_a0_59_PMLnum40
        #print(name)
        key=(infos[1],infos[2],infos[3])
        isY = True if "ot" in key else False
        return isY,key,name

    def load_files(self):
        points,list_of_files = {}, glob.glob(f"{self.data_path}*")
        nFile = len(list_of_files)
        X_,Y_ = pd.DataFrame(),pd.DataFrame()
        for id,file in enumerate(list_of_files) :
            assert os.path.isfile(file)
            isY,key,name = self.file2key(file)
            if not "right" in key:
                continue
            with open(file, 'r') as file :
                filedata = file.read()           
            filedata = StringIO(filedata.replace('D', 'e'))     #真麻烦
            df = pd.read_csv(filedata,header = None,dtype=np.float)    
            columns = df.columns
            #df.rename(columns={columns[0]:key},inplace=True) 
            #print(f"{name}={df.shape}\n{df}")       
            if isY:
                Y_[key]=df[0]
            else:
                X_[key]=df[0]
        print(f"X_={X_.shape} {X_.head()}\n{X_.tail()}")
        print(f"Y_={Y_.shape} {Y_.head()}\n{Y_.tail()}")
        num_features = X_.shape[1]
        assert X_.shape[0]==Y_.shape[0]
        self.X, self.Y = X_.values.astype('float32'), Y_.values.astype('float32')
        self.Y=self.Y[:,2]
        self.nFeature = self.X.shape[1]
        #self.nClasses = 0
        return 

    def __init__(self, dataset, data_path, normalize=False,
                 quantile_transform=False, output_distribution='normal', quantile_noise=1.0e-3, **kwargs):
        self.random_state = 42
        self.quantile_noise = quantile_noise
        self.name = f"{dataset}"
        self.data_path = data_path
        self.load_files()
        self.zero_feats=[]


if __name__ == "__main__":
    dataset = "PML"
    data = PML_dataset(dataset,data_path="E://xiada//FengNX//200Groups_For training PML//G1//")
    
    config = quantum_forest.QForest_config(data,0.002,feat_info="importance")   #,feat_info="importance"
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)

    config.model="QForest"      #"QForest"            "GBDT" "LinearRegressor"    
    nFold = 5 if dataset != "HIGGS" else 20
    folds = KFold(n_splits=nFold, shuffle=True)
    index_sets=[]
    for fold_n, (train_index, valid_index) in enumerate(folds.split(data.X)):
        index_sets.append(valid_index)
    for fold_n in range(len(index_sets)):
        config, visual = InitExperiment(config, fold_n)
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
        Fold_learning(fold_n,data,config,visual)
        break
