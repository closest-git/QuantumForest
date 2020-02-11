import torch.nn as nn
from .DecisionBlock import *

class QForest_Net(nn.Module):
    def __init__(self, in_features, config,feat_info=None):
        super(QForest_Net, self).__init__()
        self.layers = nn.ModuleList()
        self.nTree = config.nTree
        nFeat = in_features
        for i in range(config.nLayers):
            if i > 0:
                nFeat = config.nTree
                feat_info = None
            self.layers.append(
                DecisionBlock(nFeat, config, flatten_output=False,feat_info=feat_info)
            )

        self.pooling = None
        print("====== QForest_Net::__init__")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=-1)        #self.pooling(x)
        return x