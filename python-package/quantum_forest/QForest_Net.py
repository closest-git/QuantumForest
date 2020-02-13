import torch.nn as nn
import seaborn as sns;      sns.set()
import numpy as np
from .DecisionBlock import *
import matplotlib.pyplot as plt

class QForest_Net(nn.Module):
    def __init__(self, in_features, config,feat_info=None):
        super(QForest_Net, self).__init__()
        config.feat_info = feat_info
        self.layers = nn.ModuleList()
        self.nTree = config.nTree
        self.config = config
        #self.nAtt, self.nzAtt = 0, 0        #sparse attention
        nFeat = in_features
        for i in range(config.nLayers):
            if i > 0:
                nFeat = config.nTree
                feat_info = None
            self.layers.append(
                DecisionBlock(nFeat, config, flatten_output=False,feat_info=feat_info)
                #MultiBlock(nFeat, config, flatten_output=False, feat_info=feat_info)
            )

        self.pooling = None
        print("====== QForest_Net::__init__")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-1)        #self.pooling(x)
        return x

    def AfterEpoch(self, isBetter=False, epoch=0, accu=0):
        attentions=[]
        for layer in self.layers:
            #self.nAtt, self.nzAtt = self.nAtt+layer.nAtt, self.nzAtt+layer.nzAtt
            layer.AfterEpoch(epoch)
            for att in layer.get_attentions():
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


