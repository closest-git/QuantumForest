
import os, sys
import math
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

class PML_data3D(quantum_forest.TabularDataset):
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
            self.X = 2*((self.X-x_0)/(x_1-x_0)-0.5)
        if isY:
            self.Y_trans_method = ""
            y_0,y_1=self.Y.min(),self.Y.max()
            self.Y = 2*((self.Y-y_0)/(y_1-y_0)-0.5)
            # self.Y = (self.Y-y_0)/(y_1-y_0)
        lenY = np.linalg.norm(self.Y);            lenX = np.linalg.norm(self.X)

    def load_files(self,isGroup=True):
        self.isTrain = True
        self.isSample = False
        nPt=0
            
        if self.pkl_path is not None and os.path.isfile(self.pkl_path):
            print("====== PML_data3D::load_files pkl_path={} ......".format(self.pkl_path))
            with open(self.pkl_path, "rb") as fp:
                [self.X,self.Y,nPt] = pickle.load(fp)            
            gc.collect()
        else:
            listX_,listY_=[],[]
            
            if isGroup:
                group_files=[]
                for group_path in self.data_path:
                    points,files = {}, [x[0] for x in os.walk(group_path) if x[0]!=group_path]   #glob.glob(f"{self.data_path}*")  
                    group_files.extend(files)              
            else:
                group_files = self.data_path
            nMostGroup = len(group_files)
            if nMostGroup==0:
                return

            for pt_files in group_files:
                print(f"\r{nPt}\tload files@{pt_files}......",  end="")
                #if pt_files==group_path:                continue
                list_of_files = glob.glob(f"{pt_files}/*")
                if len(list_of_files)!=15:
                    continue
                X_,Y_,_ = self.load_files_v0(pt_files+"/")
                if X_ is None or Y_ is None:
                    print(f"FAILED!!!!!     {nPt}\tload files@{pt_files}...... FAILED!!!!!!")
                    continue
                ldX,ldY = X_.shape[-1], Y_.shape[-1]
                if self.isPrecit:
                    assert ldX==12 
                    Y_ = None
                else:
                    if ldX!=12 or ldY!=3:
                        print(f"FAILED!!!!!     {nPt}\tload files@{pt_files}...... ldX={ldX} ldY={ldY}!!!!!!")
                        continue
                    assert ldX==12 and ldY==3
                listX_.append(X_);      listY_.append(Y_)
                nPt = nPt+1
                if nPt>=self.nMostPt:   
                    break
            if len(listX_)>0:            
                self.X = np.vstack(listX_)
                if not self.isPrecit:
                    self.Y = np.vstack(listY_)
                    if self.ex_ey_hz>0 and self.ex_ey_hz<3:
                        self.Y=self.Y[:,self.ex_ey_hz]                
                else:
                    self.Y = np.zeros((self.X.shape[0],3))
                self.nFeature = self.X.shape[1]
            else:
                print(f"Failed to load data@{self.data_path}!!!")
            if self.pkl_path is not None:
                with open(self.pkl_path, "wb") as fp:
                    pickle.dump([self.X,self.Y,nPt], fp)
        self.nFeature = self.X.shape[1]
        nX,nY=len(self.X), len(self.Y)
        lenY = np.linalg.norm(self.Y)
        lenX = np.linalg.norm(self.X)
        # if self.isScale:
        #     x_0,x_1=self.X.min(),self.X.max()
        #     self.X = (self.X-x_0)/(x_1-x_0)
        #     y_0,y_1=self.Y.min(),self.Y.max()
        #     self.Y = (self.Y-y_0)/(y_1-y_0)
        #     lenY = np.linalg.norm(self.Y);            lenX = np.linalg.norm(self.X)
        if False:   #时间太长，pandas_profiling低效
            df = pd.DataFrame(self.X)
            print(df.head())
            pfr = pandas_profiling.ProfileReport(df)
            pfr.to_file("./example.html")
        self.nPoint = nPt
        print(f"X_={self.X.shape} |X|={lenX:.4g} {lenX/nX:.4g}\t \n{self.X[:5,:]}\n{self.X[-5:,:]} ")
        print(f"Y_={self.Y.shape} ex_ey_hz={self.ex_ey_hz}\t |Y|={lenY:.4g} {lenY/nY:.4g}\t \n{self.Y[:10]}\n{self.Y[-10:]}")
        return 
    
    def plot_arr(self,arr,title):
        # title = title + str(arr.shape)
        nFeat = 1 if len(arr.shape)==1 else math.sqrt(arr.shape[1])
        df = pd.DataFrame(arr)
        df.hist(figsize=(9*nFeat,6*nFeat), bins = 100,ec="k", log=True)
        plt.tight_layout()
        plt.suptitle(f"Histogram of {title} data shape={arr.shape}")
        # plt.show()   
        plt.savefig(f"F:/Datasets/PML3D/{self.out_name_}_{title}.png")     
        plt.close()

    def plot(self,tit):
        if self.nPoint>0:
            self.plot_arr(self.X,f"X@{tit}")
            self.plot_arr(self.Y,f"Y@{tit}")

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
        if X_.shape[0]!=Y_.shape[0]:
            return None,None,None
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
        #2028组 0.009/0.237
        self.random_state = 42
        self.quantile_noise = quantile_noise
        self.name = f"{dataset}"
        self.data_path = data_path
        self.nMostPt = nMostPt
        self.isPrecit = isPrecit
        self.ex_ey_hz = ex_ey_hz         #output component
        P__ =  data_path[0].replace('\\','_').replace('/','_').replace(':','_')        
        self.out_name_ = f"Q_{quantile_noise}_Y{self.ex_ey_hz}_P{P__}_"
        self.pkl_path = f"F:/Datasets/PML3D/{self.out_name_}.pickle"   
        self.nPoint = 0
        self.X = None;      self.Y = None
        self.load_files(isGroup=True)
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
    dataset = "PML3D"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to pretrained model', default='E:/xiada/FengNX/pml_trained.pth')
    parser.add_argument('--predict', type=str, help='the directory include testing sample files', 
            default="E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型上边的全部离散P点/")
    args, unknown = parser.parse_known_args()
      
    if False and hasattr(args,'predict') and os.path.isdir(args.predict):
        predict_(args)
        exit(-1)

    data_paths = [
            # "E:/xiada/FengNX/228组上下左右不同p点对应的十五个场分量的数值变化/模型上边的全部离散P点/",
            # "F:/PML_datas/2028组/模型上边的全部离散P点/",       #
            # "F:/PML_datas/tiny/",
            "F:/PML_datas/3D/",
            # "F:/PML_datas/3D/Z_1第一层/模型上边的全部离散P点/"
        ]
    data = PML_data3D(dataset,data_path=data_paths,ex_ey_hz=4)        #,nMostPt=10
    if data.nPoint==0:
        sys.exit(-2)
    Y_scaling=False         #isY=True
    data.Scaling(isY=Y_scaling)     
    data.plot(f"scaling={Y_scaling}")
    config = quantum_forest.QForest_config(data,0.002,feat_info="importance")   #,feat_info="importance" 

    data_root = "F:/Datasets/"   #args.data_root
    random_state = 42
    config.device = quantum_forest.OnInitInstance(random_state)
    config.err_relative = True
    config.model="QForest"      #"QForest"            "GBDT" "LinearRegressor"  
    # config.response_dim = 1
    # config.lr_base = 0.0001
    # config.path_way = "OBLIVIOUS_map"
    # config.batch_size = 1024
    # config.nTree = 1
    
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
        pkl_ = f"{data_root}{dataset}/FOLD_{fold_n}_PT{data.nPoint}_X{data.X.shape}_Y{data.ex_ey_hz}_.pickle"
        pkl_ = None
        data.onFold(fold_n,config,train_index=train_index, valid_index=valid_index,test_index=index_sets[fold_n],pkl_path=pkl_)
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
        break
