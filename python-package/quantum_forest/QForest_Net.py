'''
@Author: Yingshi Chen
@Date: 2020-02-14 11:06:23
@LastEditTime: 2020-02-24 13:25:38
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \QuantumForest\python-package\quantum_forest\QForest_Net.py
'''
import torch.nn as nn
import seaborn as sns;      sns.set()
import numpy as np
from .DecisionBlock import *
import matplotlib.pyplot as plt
from .some_utils import *
from cnn_models import *

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
        #self.init_weights()    #引起振荡，莫名其妙
        print(self)
    
    def forward(self, x):
        out = self.features(x)        
        out = out.view(out.size(0), -1)
        if hasattr(self,"linear"):
            out = self.linear(out)
        else:
            out = out.mean(dim=-1)
        return out

            
class QForest_Net(nn.Module):
    def pick_cnn(self):
        self.back_bone = 'resnet18_x'
        in_channel = self.config.response_dim
        # model_name='dpn92'
        # model_name='senet154'
        # model_name='densenet121'
        # model_name='alexnet'
        # model_name='senet154'
        #cnn_model = ResNet_custom([2,2,2,2],in_channel=1,out_channel=1)          ;#models.resnet18(pretrained=True)
        cnn_model = Simple_VGG([2,2,2,2],in_channel=in_channel,out_channel=1) 
        return cnn_model

    def __init__(self, in_features, config,feat_info=None,visual=None):
        super(QForest_Net, self).__init__()
        config.feat_info = feat_info
        self.layers = nn.ModuleList()
        self.nTree = config.nTree
        self.config = config
        #self.gates_cp = nn.Parameter(torch.zeros([1]), requires_grad=True)
        self.reg_L1 = 0.
        self.reg_L2 = 0.
        if config.leaf_output == "distri2CNN": 
            self.cnn_model = self.pick_cnn()    
        
        #self.nAtt, self.nzAtt = 0, 0        #sparse attention
        if self.config.data_normal=="NN":       #将来可替换为deepFM
            self.emb_dims = [in_features,256]   #multilayer未见效果     0.590(差于0.569)
            self.emb_dims = [in_features,128]
            #self.embedding = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_szs])
            nEmb = len(self.emb_dims)-1
            self.embeddings = nn.ModuleList(
                [nn.Linear(self.emb_dims[i], self.emb_dims[i + 1]) for i in range(nEmb)] ) 
            nFeat = self.emb_dims[nEmb]
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
        print("====== QForest_Net::__init__   OK!!!")        
        #print(self)
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
            else:
                x = x.mean(dim=-1)        #self.pooling(x)
            #x = torch.max(x,dim=-1).values
            return x
    
    def Regularization(self):
        dict_val = self.get_variables({"attention":[],"gate_values":[]})
        reg,l1,l2 = 0,0,0
        for att in dict_val["attention"]:
            a = torch.sum(torch.abs(att))/att.numel()
            l1 = l1+a
        self.reg_L1 = l1
        reg = self.reg_L1*self.config.reg_L1
        for gate_values in dict_val["gate_values"]:
            a = torch.sum(torch.pow(gate_values, 2))/gate_values.numel()
            l2 = l2+a
        self.reg_L2 = l2     
        #if self.config.reg_Gate>0:            reg = reg+self.reg_L2*self.config.reg_Gate 
        #return reg
        return reg
    
    def get_variables(self,var_dic):
        for layer in self.layers:
            var_dic = layer.get_variables(var_dic)
                #attentions.append(att)
        #all_att = torch.cat(attentions)
        return var_dic

    def AfterEpoch(self, isBetter=False, epoch=0, accu=0):
        #trainer.opt = Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), **optimizer_params)
        return

        attentions=[]
        for layer in self.layers:
            #self.nAtt, self.nzAtt = self.nAtt+layer.nAtt, self.nzAtt+layer.nzAtt
            layer.AfterEpoch(epoch)
            for att,_ in layer.get_variables():
                attentions.append(att.data.detach().cpu().numpy())
        attention = np.concatenate(attentions)  #.reshape(-1)
        self.nAtt = attention.size  # sparse attention
        self.nzAtt = self.nAtt - np.count_nonzero(attention)
        print(f"\t[nzAttention={self.nAtt} zeros={self.nzAtt},{self.nzAtt * 100.0 / self.nAtt:.4f}%]")
        #plt.hist(attention)    #histo不明显
        if self.config.plot_attention:
            nFeat,nCol = attention.shape[0],attention.shape[1]
            nCol = min(nFeat*3,attention.shape[1])
            cols = random.choices(population = list(range(attention.shape[1])),k = nCol)
            type="" if self.config.feat_info is None else "sparse"
            if self.visual is not None:
                path = f"{self.config.data_set}_{type}_{epoch}"
                params = {'title':f"{epoch} - {accu:.4f}",'cmap':sns.cm.rocket}
                self.visual.image(path,attention[:,cols],params=params)
            else:
                plt.imshow(attention[:,cols])
                plt.grid(b=None)
                plt.show()
        return
    
    def freeze_some_params(self,freeze_info):
        for layer in self.layers:
            layer.freeze_some_params(freeze_info)



