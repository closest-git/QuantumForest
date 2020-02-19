'''
@Author: your name
@Date: 2020-02-19 09:32:50
@LastEditTime: 2020-02-19 09:33:13
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \QuantumForest\python-package\quantum_forest\LinearRegressor.py
'''
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge,ElasticNet
import time

class Linear_Regressor:
    def __init__(self,params,  **kwargs):
        super(Linear_Regressor, self).__init__()
        self.alpha = 1
        self.gressor = None;        self.alg="None"
        if 'cascade' in params:
            self.alg = params['cascade']
            if params['cascade']=="lasso":
                self.gressor = Lasso(alpha=self.alpha, normalize=True)                
            elif params['cascade']=="ridge":
                self.gressor = Ridge(alpha=0.05, normalize=True)
            elif params['cascade']=="ElasticNet":
                self.gressor = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
        self.mse = 0

    def fit(self,train_set,eval_set):
        if self.gressor is None:
            return False,None,None
        t0=time.time()
        print(f"====== Linear_Regressor::Fit@{self.gressor} alpha={self.alpha}")
        x_train, y_train = train_set
        self.gressor.fit(x_train, y_train)
        pred = self.gressor.predict(x_train)
        train_mse = np.mean((pred - y_train)**2)

        y_eval = None
        if (eval_set is not None and len(eval_set) > 0):
            X_eval, y_eval = eval_set[0]
            y_pred = self.gressor.predict(X_eval)
            valid_mse = np.mean((y_pred - y_eval)**2)    #error_rate = ((y_test - prediction) ** 2).mean()
            print(f"====== {self.alg}_Regressor::fit train_mse={train_mse:.6f} valid_mse={valid_mse:.6f} Time={time.time()-t0:.2g}")
        return valid_mse,train_mse

    def BeforeFit(self,train_set,eval_sets):
        if self.gressor is None:
            return False,None,None
        y_New=[]
        t0 = time.time()
        print(f"====== Linear_Regressor::BeforeFit@{self.gressor} alpha={self.alpha}")
        x_train, y_train = train_set
        self.gressor.fit(x_train, y_train)
        pred = self.gressor.predict(x_train)
        self.mse = np.mean((pred - y_train)**2)
        y_train = y_train - pred
        y_New.append(y_train)

        if eval_sets is not None:
            for eset in eval_sets:
                t0 = time.time()
                X_eval, y_eval = eset
                pred_1 = self.gressor.predict(X_eval)
                mse = np.mean((pred_1 - y_eval)**2)
                print(f"====== {self.alg}_Regressor::fit X_eval={X_eval.shape} valid_mse={mse:.6f} Time={time.time()-t0:.2g}")
                y_eval = y_eval-pred_1
                y_New.append(y_eval)        
           
        return y_New

    def AfterPredict(self,X_,Y_):
        t0 = time.time()
        if self.gressor is not None:
            y_pred = self.gressor.predict(X_)
            Y_= Y_+y_pred
        print(f"====== {self.alg}_AfterPredict::fit X_eval={X_.shape} Time={time.time()-t0:.2g}")
        return Y_
