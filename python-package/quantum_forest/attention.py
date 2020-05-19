'''
@Author: Yingshi Chen

@Date: 2020-05-17 20:42:48
@
# Description: 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class se_reponse(nn.Module):
    def __init__(self, nTree, reduction=16):
        super(se_reponse, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nTree = nTree
        self.reduction = reduction
        self.nEmbed = max(2,self.nTree//reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(nTree, self.nEmbed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.nEmbed, nTree, bias=False),
            #nn.Sigmoid()
            nn.Softmax()
        )
    
    def forward(self, x):
        b, t, _ = x.size()
        y = torch.mean(x,dim=-1)
        y = self.fc(y)
        out = torch.einsum('btr,bt->btr', x,y) 
        # dist = torch.dist(out,out_0,2)
        # assert dist==0
        return out

class eca_reponse(nn.Module):
    def __init__(self, nTree, k_size=3):
        super(eca_reponse, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nTree = nTree
        self.k_size = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    
    def forward(self, x):
        b, t, _ = x.size()
        y = torch.mean(x,dim=-1)
        # y0 = y.unsqueeze(-1).transpose(-1, -2)
        # y0 = self.conv(y0)
        y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        y = F.sigmoid(y)
        #y = F.softmax(y)
        out = torch.einsum('btr,bt->btr', x,y) 
        return out

class eca_input(nn.Module):
    def __init__(self, nFeat, k_size=3):
        super(eca_input, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nFeat = nFeat
        self.k_size = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    
    def forward(self, x):
        b, f = x.size()
        #y = torch.mean(x,dim=0)
        y = x
        y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        y = F.sigmoid(y)
        #y = F.softmax(y)
        out = torch.einsum('bf,bf->bf', x,y) 
        return out

class se_input(nn.Module):
    def __init__(self, nFeat, reduction=4):
        super(se_input, self).__init__()
        self.nFeat = nFeat
        self.reduction = 1
        self.nEmbed = max(2,self.nFeat//reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(self.nFeat, self.nEmbed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.nEmbed, self.nFeat, bias=False),
            #nn.Sigmoid()
            nn.Softmax()
        )
    
    def forward(self, x):
        b, f = x.size()
        #y = torch.mean(x,dim=0)
        y = self.fc(x)
        out = torch.einsum('bf,bf->bf', x,y) 
        return out
    
