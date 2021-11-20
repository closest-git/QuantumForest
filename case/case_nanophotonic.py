
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
import pandas as pd
import gc
#import pandas_profiling

class NANO_data3D(quantum_forest.TabularDataset):
    def file2key(self,file):
        name = os.path.basename(file)
        infos = name.split("_")
        #Ey ex(分量)    it1(坐标)   right(上，下，左，右)   Groupth1_a0_59_PMLnum40
        #print(name)
        key=(infos[1],infos[2],infos[3])
        isY = True if "ot" in key else False
        return isY,key,name

    def problem(self):
        #return "regression"
        return "regression_N"
    
    def Scaling(self,isX=True,isY=True,method="01"):
        print(f"====== Scaling ...... method={method} onX={isX}  onY={isY}")
        if isX:
            x_0,x_1=self.X.min(),self.X.max()
            self.X = (self.X-x_0)/(x_1-x_0)
        if isY:
            y_0,y_1=self.Y.min(),self.Y.max()
            self.Y = (self.Y-y_0)/(y_1-y_0)
        lenY = np.linalg.norm(self.Y);            lenX = np.linalg.norm(self.X)
        pass
  
    def plot_arr(self,arr,title):
        title = title + str(arr.shape)
        df = pd.DataFrame(arr)
        df.hist(figsize=(9,6), bins = 100,ec="k")

        # if len(df.columns)==1:
        #     df.hist()
        # else:
        #     nRow = max(1,len(df.columns)//3)
        #     fig, axes = plt.subplots(nRow, 3, figsize=(12, 24))
        #     i = 0
        #     for triaxis in axes:
        #         for axis in triaxis:
        #             df.hist(column = df.columns[i], bins = 50, ax=axis)
        #             i = i+1
        #     # [x.title.set_size(32) for x in fig.ravel()]
        plt.tight_layout()
        plt.title(title)
        plt.show()        
        plt.close()

    def plot(self):
        if self.nPoint>0:
            self.plot_arr(self.X,"X_")
            self.plot_arr(self.Y,"Y_")

    def __init__(self, dataset, data_path, normalize=False,nMostPt=100000,isPrecit=False,
                 quantile_transform=False, output_distribution='normal', quantile_noise=1.0e-3, **kwargs):
        self.isScale = False
        self.random_state = 42
        self.quantile_noise = quantile_noise
        self.name = f"{dataset}"
        self.data_path = data_path
        self.nMostPt = nMostPt
        self.isPrecit = isPrecit
        
        P__ =  data_path[0].replace('\\','_').replace('/','_').replace(':','_')
        self.pkl_path = data_path   
        self.nPoint = 0
        self.X = None;      self.Y = None
        
        if self.pkl_path is not None and os.path.isfile(self.pkl_path):
            print("====== NanoPhotonic::load_files pkl_path={} ......".format(self.pkl_path))
            with open(self.pkl_path, "rb") as fp:
                [self.X,self.Y,self.nPoint] = pickle.load(fp)            
            gc.collect()
            self.nFeature = self.X.shape[1]
            self.isTrain = True
            self.isSample = False

        self.zero_feats=[]  
            
        

def predict_(args):
    model = args.model
    if not os.path.isfile(args.model):
        print(f"'{args.model}' is not a valid MODEL file!!!")
        return
    data_paths = [args.predict]
    data = PML_data3D("PML_predict",data_path=data_paths,nMostPt=1,isPrecit=True)
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
    dataset = "NanoPhotonic"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to pretrained model', default='E:/xiada/FengNX/pml_trained.pth')
    parser.add_argument('--predict', type=str, help='the directory include testing sample files', 
            default="")
    args, unknown = parser.parse_known_args()
      
    if False and hasattr(args,'predict') and os.path.isdir(args.predict):
        predict_(args)
        exit(-1)

    data_paths = [
            "F:/Datasets/NANO/nanop2D_600_.pickle"
        ]
    data = NANO_data3D(dataset,data_path=data_paths[0])        #,nMostPt=10
    if data.nPoint==0:
        sys.exit(-2)
    # data.Scaling(isY=False)
    # data.plot()
    config = quantum_forest.QForest_config(data,0.002,feat_info="importance")   #,feat_info="importance"  
    

    data_root = "F:/Datasets/"   #args.data_root
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)
    config.err_relative = True
    config.model="QForest"      #"QForest"            "GBDT" "LinearRegressor"  
    config.response_dim = 5
    # config.lr_base = 0.0001
    #config.path_way = "OBLIVIOUS_map"
    # config.batch_size = 64
    # config.nTree = 1
    config.nMostEpochs = 100
    fold_n = 0
    config, visual =  quantum_forest.InitExperiment(config, 0)
    # config.response_dim = 3
    config.feat_info = None
    nTrain = (int)(data.nPoint/6*5);    nTest=data.nPoint-nTrain
    train_index=np.arange(nTrain)
    test_index=nTrain+np.arange(nTest)
    valid_index=test_index
    print(f"train={len(train_index)} valid={len(valid_index)} test={len(test_index)}")
    pkl_ = None
    data.onFold(fold_n,config,train_index=train_index, valid_index=valid_index,test_index=test_index,pkl_path=pkl_)
    # data.plot()
    data.Y_mean,data.Y_std = data.y_train.mean(), data.y_train.std()
    if False:
        Fold_learning(fold_n,data,config,visual)
    else:     
        config.in_features = data.X_train.shape[1]
        config.tree_module = quantum_forest.DeTree       
        learner = quantum_forest.QuantumForest(config,data,feat_info=None,visual=visual).   \
            fit(data.X_train, data.y_train, eval_set=[(data.X_valid,data.y_valid)])
        best_score = learner.best_score,0
        
