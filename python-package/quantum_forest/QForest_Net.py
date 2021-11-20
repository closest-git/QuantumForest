'''
@Author: Yingshi Chen

@Date: 2020-02-14 11:06:23
@
# Description: 
'''

import torch.nn as nn
import seaborn as sns;      sns.set()
import numpy as np
import time
import gc
from .DecisionBlock import *
import matplotlib.pyplot as plt
from .some_utils import *
from cnn_models import *
from qhoptim.pyt import QHAdam
from .experiment import *
from .Visualizing import *

class Simple_CNN(nn.Module):
    def __init__(self, num_blocks, in_channel=3,out_channel=10):
        super(Simple_CNN, self).__init__()
        self.in_planes = 64
        nK=128
        self.conv1 = nn.Conv2d(in_channel, nK, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nK)
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.linear = nn.Linear(1024, out_channel) 有意思
    
    def forward(self, x):
        if False:    #仅用于调试
            shape=x.shape
            x=x.view(shape[0],1,shape[1],shape[2])
            x=x.view(shape[0],shape[1],shape[2])
            x = torch.max(x,dim=-1).values
            x = x.mean(dim=-1) 
            return x
        else:        
            out = F.relu(self.bn1(self.conv1(x)))     
            #out = self.bn1(self.conv1(x))       
            out = F.avg_pool2d(out, 8)   
            #out = F.max_pool2d(out, 8)          
            out = out.view(out.size(0), -1)
            if hasattr(self,"linear"):
                out = self.linear(out)
            else:
                out = out.mean(dim=-1)
                #out = torch.max(out,-1).values 略差
            return out

class Simple_VGG(nn.Module):
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def _make_layers(self, cfg, in_channel=3):
        layers = []
        in_channels = in_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           #nn.ReLU(inplace=True)
                           ]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def __init__(self, num_blocks, in_channel=3,out_channel=10):
        super(Simple_VGG, self).__init__()
        self.cfg = {
            'VGG_1': [64,'M'],
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        self.type = 'VGG_1'
        self.features = self._make_layers(self.cfg[self.type],in_channel=in_channel)
        self.out_channel = out_channel
        #self.init_weights()    #引起振荡，莫名其妙
        if False and self.out_channel>1:
            self.linear = nn.Linear(4096, self.out_channel)
            nn.init.kaiming_normal_(self.linear.weight)
        print(self)
    
    def forward(self, x):
        out = self.features(x)        
        out = out.view(out.size(0), -1)
        if hasattr(self,"linear"):
            out = self.linear(out)
        else:            
            if self.out_channel>1:   #"classification":
                out = out.view(out.size(0), -1,self.out_channel)
                assert len(out.shape)==3
                out = out.mean(dim=-2)
            else:
                out = out.mean(dim=-1)
        return out

            
class QF_Net(nn.Module):
    def pick_cnn(self):
        self.back_bone = 'resnet18_x'
        in_channel = self.config.response_dim
        if self.config.problem()=="classification":
            out_channel = self.config.response_dim 
        else:
            out_channel = 1
        # model_name='dpn92'
        # model_name='senet154'
        # model_name='densenet121'
        # model_name='alexnet'
        # model_name='senet154'
        #cnn_model = ResNet_custom([2,2,2,2],in_channel=1,out_channel=1)          ;#models.resnet18(pretrained=True)
        cnn_model = Simple_VGG([2,2,2,2],in_channel=in_channel,out_channel=out_channel) 
        return cnn_model

    def __init__(self, in_features, config,feat_info=None,visual=None):
        super(QF_Net, self).__init__()
        config.feat_info = feat_info
        self.layers = nn.ModuleList()
        self.nTree = config.nTree
        self.config = config
        #self.gates_cp = nn.Parameter(torch.zeros([1]), requires_grad=True)
        self.reg_L1 = 0.
        self.L_gate = 0.
        if config.leaf_output == "distri2CNN": 
            self.cnn_model = self.pick_cnn()    
        
        #self.nAtt, self.nzAtt = 0, 0        #sparse attention
        if self.config.data_normal=="NN":       #将来可替换为deepFM
            nEmbed = max(2,in_features//2)
            nEmbed = in_features*2,256,128
            self.emb_dims = [in_features,nEmbed]
            #self.embedding = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_szs])
            nLayer = len(self.emb_dims)
            self.embeddings = nn.ModuleList(
                [nn.Linear(self.emb_dims[i], self.emb_dims[i + 1]) for i in range(nLayer-1)] ) 
            nFeat = self.emb_dims[nLayer-1]
            for layer in self.embeddings:
                nn.init.kaiming_normal_ (layer.weight.data)
        else:
            self.embeddings = None
            nFeat = in_features
        for i in range(config.nLayers):
            if i > 0:
                nFeat = config.nTree
                feat_info = None
            hasBN = config.data_normal == "BN" and i > 0
            self.layers.append(
                DecisionBlock(nFeat, config,hasBN=hasBN, flatten_output=False,feat_info=feat_info)
                #MultiBlock(nFeat, config, flatten_output=False, feat_info=feat_info)
            )

        self.pooling = None
        self.visual = visual
        #self.nFeatInX = nFeat
        print("====== QF_Net::__init__   OK!!!")        
        #print(self)
        if True:
            dump_model_params(self)
    

  
    def forward(self, x):
        nBatch = x.shape[0]
        if self.config.feature_fraction<1:
            x=x[:,self.config.trainer.feat_cands]
            
        if self.embeddings is not None:
            for layer in self.embeddings:
                x = layer(x)
        for layer in self.layers:
            x = layer(x)
            
        self.Regularization()       #统计Regularization
        if self.config.leaf_output == "distri2CNN": 
            out = self.cnn_model(x)
            #x1 = x.contiguous().view(nBatch, -1).mean(dim=-1)  #可惜效果不明显
            #out += x1
            return out
        else:
            if self.config.problem()=="classification":
                assert len(x.shape)==3
                x = x.mean(dim=-2)
            elif self.config.problem()=="regression_N":
                pass
            else:
                x = x.mean(dim=-1)        #self.pooling(x)
            #x = torch.max(x,dim=-1).values
            return x
    
    def Regularization(self):
        # self.reg_L1,self.reg_L2=0,0
        # reg = 0
        # return
        
        dict_val = self.get_variables({"attention":[],"gate_values":[]})
        reg,l1,l2 = 0,0,0
        for att in dict_val["attention"]:
            a = torch.sum(torch.abs(att))/att.numel()
            l1 = l1+a
        self.reg_L1 = l1
        reg = self.reg_L1*self.config.reg_L1
        for gate_values in dict_val["gate_values"]:
            if self.config.support_vector=="0": 
                a = torch.sum(torch.pow(gate_values, 2))/gate_values.numel()
            else:
                gmin,gmax = torch.min(gate_values).item(),torch.max(gate_values).item()
                a = torch.sum(gate_values)/gate_values.numel()
            l2 = l2+a
        self.L_gate = l2     
        #if self.config.reg_Gate>0:            reg = reg+self.L_gate*self.config.reg_Gate 
        #return reg
        return reg
    
    def get_variables(self,var_dic):
        for layer in self.layers:
            var_dic = layer.get_variables(var_dic)
                #attentions.append(att)
        #all_att = torch.cat(attentions)
        return var_dic

    def AfterEpoch(self, isBetter=False, epoch=0, accu=0):
        attentions=[]
        if False:            
            for layer in self.layers:
                #self.nAtt, self.nzAtt = self.nAtt+layer.nAtt, self.nzAtt+layer.nzAtt
                layer.AfterEpoch(epoch)
                for att,_ in layer.get_variables():
                    attentions.append(att.data.detach().cpu().numpy())
        else:
            dict_val = self.get_variables({"attention":[]})
            for att in dict_val["attention"]:
                attentions.append(att.data.detach().cpu().numpy())
        attention = np.concatenate(attentions)  #.reshape(-1)
        self.nAtt = attention.size  # sparse attention
        self.nzAtt = self.nAtt - np.count_nonzero(attention)
        #print(f"\t[nzAttention={self.nAtt} zeros={self.nzAtt},{self.nzAtt * 100.0 / self.nAtt:.4f}%]")
        #plt.hist(attention)    #histo不明显
        if self.config.plot_attention:
            nFeat,nCol = attention.shape[0],attention.shape[1]
            nCol = min(nFeat*3,attention.shape[1])
            cols = random.choices(population = list(range(attention.shape[1])),k = nCol)
            type="" if self.config.feat_info is None else "sparse"
            if self.visual is not None:
                path = f"{self.config.data_set}_{type}_{epoch}"
                params = {'title':f"{epoch} - {accu:.4f}",'cmap':sns.cm.rocket}
                # self.visual.image(path,attention[:,cols],params=params)
            else:
                pass
                # plt.imshow(attention[:,cols])
                # plt.grid(b=None)
                # plt.show()
        return
    
    def freeze_some_params(self,freeze_info):
        for layer in self.layers:
            layer.freeze_some_params(freeze_info)

'''
    sklearn-style 和QForest_Net区别很大
'''
class QuantumForest(object):
    def load_dll(self):
        pass

    def __init__(self, config,data,feat_info=None,visual=None):
        super(QuantumForest, self).__init__()
        self.config = config
        self.problem = None
        self.preprocess = None
        self.Learners=[]
        in_features = config.in_features
        qForest = QF_Net(in_features,config, feat_info=feat_info,visual=visual).to(config.device)   
        self.Learners.append(qForest)    
        self._n_classes = None
        self.best_iteration_ = 0
        self.best_iteration = 0
        self.best_score = 0
        self.visual = visual

        #weight_decay的值需要反复适配       如取1.0e-6 还可以  0.61142-0.58948
        self.optimizer=QHAdam;           
        self.optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998),'lr':config.lr_base,'weight_decay':1.0e-8 }
        #一开始收敛快，后面要慢一些
        #optimizer = torch.optim.Adam;    optimizer_params = {'lr':config.lr_base }
        config.eval_batch_size = 512 if config.leaf_output=="distri2CNN" else \
                512 if config.path_way=="TREE_map" else 1024

        wLearner=self.Learners[-1]
        self.trainer = Experiment(
            config,data,
            model=wLearner, loss_function=F.mse_loss,
            experiment_name=config.experiment,
            warm_start=False,   
            Optimizer=self.optimizer,        optimizer_params=self.optimizer_params,
            verbose=True,      #True
            n_last_checkpoints=5,
        )   
        config.trainer = self.trainer        
        self.trainer.SetLearner(wLearner)
        print(f"====== QuantumForest__init__ config={config}\n")

    def __del__(self):
        try:
            #print("LiteMORT::__del__...".format())
            if self.hLIB is not None:
                self.mort_clear(self.hLIB)
            self.hLIB = None
        except AttributeError:
            pass
    
    def EDA(self,flag=0x0):
        '''

        :param flag:
        :return:
        '''
        return
    
    def fit(self,X_train_0, y_train,eval_set,categorical_feature=None,discrete_feature=None, flags={'report_frequency':1000}):
        assert len(eval_set)==1
        X_eval_0,y_valid = eval_set[0]
        config = self.config
        print("====== QuantumForest_fit X_train_0={} y_train={}......".format(X_train_0.shape, y_train.shape))
        gc.collect()
        # isUpdate,y_train_1,y_eval_update = self.problem.BeforeFit([X_train_0, y_train], eval_set)
        # if isUpdate:
        #     y_train=y_train_1
        self.feat_info={"categorical":categorical_feature,"discrete":discrete_feature}
        in_features = X_train_0.shape[1]

        if False:       # trigger data-aware init,作用不明显
            with torch.no_grad():
                res = qForest(torch.as_tensor(data.X_train[:1000], device=config.device))
        #if torch.cuda.device_count() > 1:        model = nn.DataParallel(model)

        loss_history, mse_history = [], []
        best_mse,best_epoch_ = float('inf'),float('inf')
        best_step_mse = 0
        early_stopping_rounds = 3000
        early_stopping_epochs = 2
        report_frequency = flags['report_frequency']
        trainer = self.trainer
        data = self.trainer.data
        wLearner=self.Learners[-1]        

        print(f"======  trainer.learner={trainer.model}\ntrainer.opt={trainer.opt}"\
            f"\n======  config={config.__dict__}")
        print(f"======  X_train={data.X_train.shape},YY_train={y_train.shape}")
        print(f"======  |YY_train|={np.linalg.norm(y_train):.3f},mean={y_train.mean():.3f} std={y_train.std():.3f}")
        wLearner.AfterEpoch(isBetter=True, epoch=0)
        epoch,t0=0,time.time()
        nBatch = (int)(data.__len__()*config.bagging_fraction)//config.batch_size
        nBatch = min(1024,nBatch)
        for epoch in range(config.nMostEpochs):   
            config.log_writer.epoch = epoch
            # for batch in iterate_minibatch(data.X_train, y_train, batch_size=config.batch_size,shuffle=True, epochs=float('inf')):
            for batch in data.yield_batch(config.batch_size,subsample=config.bagging_fraction,shuffle=True, nMostBatch=nBatch):
                config.log_writer.step = trainer.step
                metrics = trainer.train_on_batch(*batch, device=config.device)                            
                loss_history.append(metrics['loss'])
                if trainer.step%10==0 or (trainer.step)%nBatch==0:
                    symbol = "^" if config.cascade_LR else ""
                    print(f"\r============ {trainer.step} {trainer.step-epoch*nBatch}{symbol}/{nBatch}\t{metrics['loss']:.5f}\tL1=[{wLearner.reg_L1:.4g}*{config.reg_L1}]"
                    f"\tL2=[{wLearner.L_gate:.4g}*{config.reg_Gate}]\ttime={time.time()-t0:.2f}\t"
                    ,end="")
                
            if torch.cuda.is_available():   torch.cuda.empty_cache()
            mse = trainer.AfterEpoch(epoch,data.X_valid,y_valid,best_mse)            
            if mse < best_mse:
                best_mse,best_epoch_ = mse,epoch
                trainer.save_checkpoint(tag='best_mse')
            mse_history.append(mse)
            if config.average_training:
                trainer.load_checkpoint()  # last
                trainer.remove_old_temp_checkpoints()
            if self.visual is not None:
                self.visual.UpdateLoss(title=f"Accuracy on \"{data.name}\"",legend=f"{config.experiment}", loss=mse,yLabel="Accuracy")
            print(f"Time: [{time.time()-t0:.6f}]	[{epoch}]	valid_0's rmse: {mse}")   #为了和其它库的epoch输出一致
            # VisualAfterEpoch(epoch,visual,config,mse)  
            if config.verbose:
                config.log_writer.add_scalar('train/loss', metrics['loss'], epoch)
                config.log_writer.add_scalar('valid/rmse', mse, epoch)
            if "test_once" in flags:
                break    

            if epoch > best_epoch_ + early_stopping_epochs:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print(f"\tBest epoch: {best_epoch_} Best Val MSE: {best_mse:.5f}" )                  
                break
            
        
        # if data.X_test is not None:   外层更合理
        #     YY_test = data.y_test
        #     mse = trainer.AfterEpoch(epoch,data.X_test, YY_test,best_mse,isTest=True)             
        #     best_mse = mse
        #     trainer.save_checkpoint(tag=f'last_{best_mse:.6f}')
        self.best_score = best_mse        # return best_mse,mse
        return self


    def predict(self, X_,pred_leaf=False, pred_contrib=False,raw_score=False,num_iteration=-1, flag=0x0):
        """Predict class or regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        self.profile.Snapshot("PRED_0")
        # print("====== LiteMORT_predict X_={} ......".format(X_.shape))
        Y_ = self.predict_raw(X_,pred_leaf, pred_contrib,raw_score,flag=flag)
        Y_ = self.problem.AfterPredict(X_,Y_)
        Y_ = self.problem.OnResult(Y_,pred_leaf,pred_contrib,raw_score)
        return Y_


def InitExperiment(config,fold_n):
    config.experiment = f'{config.data_set}_{config.model_info()}_{fold_n}'   #'year_node_shallow'
    #experiment = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}_{:0>2d}'.format(experiment, *time.gmtime()[:5])
    #visual = quantum_forest.Visdom_Visualizer(env_title=config.experiment)
    visual = Visualize(env_title=config.experiment)
    visual.img_dir = "./results/images/"
    print("experiment:", config.experiment)
    log_path=f"logs/{config.experiment}"
    if os.path.exists(log_path):        #so strange!!!
        import shutil
        print(f'experiment {config.experiment} already exists, DELETE it!!!')
        shutil.rmtree(log_path)
    return config,visual