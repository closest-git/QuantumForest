import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import copy
from .sparse_max import sparsemax, sparsemoid, entmoid15,entmax15
from .some_utils import check_numpy
from warnings import warn

class DeTree(nn.Module):
    def init_attention(self, feat_info=None): 
        self.attention_reuse = False       #效果不理想，奇怪        
        self.nGateFuncs = self.num_trees*self.nFeature
        self.nSingle = self.in_features
        if self.attention_reuse:
            self.nGateFuncs = self.nGateFuncs//10
            print(f"======  init_attention nGate={self.nGateFuncs}")
            self.choice_map = [random.randrange(0, self.nGateFuncs) for _ in range(self.num_trees*self.nFeature)] 
            assert len(self.choice_map)==self.num_trees*self.nFeature

        weight=None
        if feat_info is not None and 'importance' in feat_info.columns:
            importance = torch.from_numpy(feat_info['importance'].values).float()
            assert importance.shape[0] == self.in_features
            fmax, fmin = torch.max(importance), torch.min(importance)
            weight = importance / fmax
            # weight[weight<1.0e-4] = 1.0e-4
            weight = weight * weight * weight
            nShow=min(self.in_features/2,20)
            print(f"====== feat weight={weight[0:nShow]}...{weight[self.in_features-nShow:self.in_features-1]} fmax={fmax}, fmin={fmin}")

        #only for self.no_attention and weight is None
        self.feat_map = random.choices(population = list(range(self.in_features)),k = self.nGateFuncs) #weights = weight,
        #self.feat_map[0]=1
        self.init_attention_func=nn.init.eye
        if self.no_attention:
            self.feat_weight = nn.Parameter(torch.Tensor(self.nGateFuncs).uniform_(), requires_grad=True)
            pass
        elif False:  #仅用于对比
            feat_val = torch.zeros([self.in_features, self.nGateFuncs])
            for i,pos in enumerate(self.feat_map): 
                feat_val[:, i] = 0
                feat_val[pos, i] = 1
            self.feat_attention = nn.Parameter(feat_val, requires_grad=True)
            print(f"===== !!! init_attention: SET {self.nGateFuncs}-featrues to 1 !!!")
        else:
            #kaiming_uniform_可以加速收敛，但最终结果差不多
            self.init_attention_func = nn.init.uniform_    #nn.init.kaiming_normal_        #nn.init.uniform_
            feat_val = torch.zeros([self.in_features, self.nGateFuncs])
            self.init_attention_func(feat_val)

            if weight is not None :
                for i in range(self.nGateFuncs):
                    # print(feat_val[:,i])
                    feat_val[:, i] = feat_val[:, i] * weight
                    # print(feat_val[:, i])
                self.feat_attention = nn.Parameter(feat_val, requires_grad=True)
                print(f"====== init_attention from feat_info={feat_info.columns}")
            self.feat_attention = nn.Parameter( feat_val, requires_grad=True )
        #self.feat_W = nn.Parameter(torch.Tensor(self.in_features).uniform_(), requires_grad=True)

        print(f"====== init_attention f={self.init_attention_func.__name__} no_attention={self.no_attention}")

    #weights computed as entmax over the learnable feature selection matrix F ∈ R d×n
    def get_attention_value(self,input):
        nBatch,in_feat = input.shape[0],input.shape[1]
        #att_max = torch.max(self.feat_attention)       无效
        #self.feat_attention.data = self.feat_attention.data / att_max
        choice_weight = self.choice_function(self.feat_attention, dim=0)        #choice_function=entmax15
        #c_max = torch.max(choice_weight)
        #choice_weight[choice_weight < c_max] = 0

        #choice_weight = torch.einsum('f,fn->fn', self.feat_W,choice_weight)
        # ^--[in_features, num_trees, depth]
        if self.attention_reuse:
            choice_weight = choice_weight[:, self.choice_map]
            #for i,id in enumerate(self.choice_map):
            #    self.choice[:,i] = choice_weight[:,id]
        else:
            pass
        if False:   #如何稀疏化?
            expand_cols=[]
            for i in range(in_feat):
                expand_cols.extend(list(range(self.num_trees)))
            input_expand = input[:,expand_cols]
            input_expand = input_expand.view(nBatch,self.num_trees,-1)
            choice_weight = choice_weight.view(self.in_features,self.num_trees,-1)
            feature_values = torch.einsum('btf,ftd->btd', input_expand, choice_weight) #x=bt
            feature_values = feature_values.view(nBatch, self.num_trees, -1)
        else:
            choice_weight = choice_weight.view(self.in_features,self.num_trees,-1)
            feature_values = torch.einsum('bf,fnd->bnd', input, choice_weight)
        return feature_values
    
    def InitPathWay(self):
        nLeaf = 2 ** self.depth  
        nNode,level_nodes,path_map=0,[],[] 
        for level in range(self.depth): #level_nodes has no Leaf!!!
            if level==0:
                nodes=[0]  
            else:
                nodes = []
                for i,node in enumerate(uppers):
                    nodes.append(nNode+2*i)
                    nodes.append(nNode+2*i+1)
            level_nodes.append(nodes)
            uppers = copy.deepcopy(nodes);     nNode+=len(nodes)
        print(f"====== DeTree::InitPathWay path_map={self.config.path_way} depth={self.depth} nLeaf={nLeaf} nGatingNode={nNode} nFeature={self.nFeature}")
        
        if self.config.path_way=="OBLIVIOUS_map":
            path_map,nodes=[],list(range(self.depth))
            for no in range(nLeaf):
                path = []
                for j in range(self.depth):                    
                    left_rigt,node = no%2,nodes[j]
                    path.append(2*node+left_rigt)      #OBLIVIOUS Tree
                    no = no//2
                path_map.extend(path)
                print(f"\t{no}\tpath={path}")
            self.path_map = nn.Parameter(torch.tensor(path_map), requires_grad=False)
            #print(f"====== DeTree::__init__ path_map={path_map}")
            #self.AfterEpoch(-1)
        elif self.config.path_way=="TREE_map":  #leaf=2^D feat=2^D-1 attention=2*(2^D-1)
            #print(f"====== DeTree::__init__ path_map={path_map}")
            nFeat = nNode            
            for no in range(nLeaf):
                path = []
                for j in reversed(range(self.depth)):                    
                    left_rigt, node = no%2,level_nodes[j][no//2]
                    path.append(2*node+left_rigt)      #OBLIVIOUS Tree
                    no = no//2
                path.reverse()
                print(f"\t{no}\tpath={path}")
                path_map.extend(path)
            assert max(path_map)<=2*nFeat-1
            self.path_map = nn.Parameter(torch.tensor(path_map), requires_grad=False)
            
        else: # binary codes for mapping between 1-hot vectors and bin indices
            with torch.no_grad():
                indices = torch.arange(2 ** self.depth)
                offsets = 2 ** torch.arange(self.depth)
                bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
                bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1).cuda()
                self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
                print("====== DeTree::__init__ bin_codes_1hot={bin_codes_1hot}")
                # ^-- [depth, 2 ** depth, 2]

    def AfterEpoch(self,epoch=0):
        if epoch==0:  
            return
        return
        if self.path_map is not None:
            path_map = self.path_map.data.cpu()       #.view(-1,self.nFeature)
            #nPath = path_map.shape[0]
            feats,f_map = list(range(self.nFeature)),{} 
            random.shuffle(feats)
            for i,no in enumerate(feats):
                f_map[2*i] = 2*no
                f_map[2*i+1] = 2*no+1
            print(f"!!!!!! path_map: {path_map[0:16]}...",end="")            
            path_map.apply_(lambda y: f_map[y])   
            self.path_map.data = path_map.cuda()
            print(f"=>{self.path_map.data[0:16]}...")            
            return


    def __init__(self, in_features, num_trees,config, flatten_output=True,
                 choice_function=sparsemax, bin_function=sparsemoid,feat_info=None,
                 initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0,
                 ):
        """
        Differentiable Sparsemax Trees. 
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param response_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, response_dim],
            by default returns [..., num_trees * response_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        """
        super().__init__()
        depth = config.depth
        self.config = config
        self.isInitFromData = False
        self.in_features = in_features
        self.depth, self.num_trees, self.response_dim, self.flatten_output = \
            depth, num_trees, config.response_dim, config.flatten_output
        
        #self.choice_function, self.bin_function = choice_function, bin_function
        self.choice_function = entmax15
        self.bin_func = "05_01"     #"05_01"        "entmoid15" "softmax"
        self.no_attention = config.no_attention
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.init_responce_func = initialize_response_
        self.gate_values = 0

        if self.config.leaf_output == "learn_distri":
            self.response = nn.Parameter(torch.zeros([num_trees, self.response_dim, 2 ** depth]), requires_grad=True)
            initialize_response_(self.response)
        else:
            self.response = None
        self.path_map = None

        self.nFeature = depth
        if self.config.path_way=="TREE_map":
            self.nFeature = 2**depth-1
        self.init_attention(feat_info)
        self.nAtt = self.feat_attention.numel()  # sparse attention
        self.nzAtt = self.nAtt - self.feat_attention.nonzero().size(0)
        
        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, self.nFeature], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)
        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, self.nFeature], float('nan'), dtype=torch.float32), requires_grad=True
        )

        self.InitPathWay()        

    def forward(self, input):
        if not self.isInitFromData:
            self.initialize(input)
            self.isInitFromData = True
        assert len(input.shape) >= 2        
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)

        batch_size = input.shape[0]
        nLeaf = 2 ** self.depth
        # new input shape: [batch_size, in_features]
        if self.no_attention:
            feature_values = input[:, self.feat_map]  # torch.index_select(input.flatten(), 0, self.feat_select)
            #feature_values = torch.einsum('bf,f->bf',feature_values,self.feat_weight)
            feature_values = feature_values.reshape(-1, self.num_trees, self.depth)
            assert feature_values.shape[0] == input.shape[0]
        else:
            feature_values = self.get_attention_value(input)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)
        #threshold_logits = (feature_values ) * torch.exp(-self.log_temperatures) - self.feature_thresholds 差不多
        if False:  #relative distance has no effect
            threshold_logits = threshold_logits/(torch.abs(self.feature_thresholds)+1.0e-4)
        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        #RESPONSE_WEIGHTS 1)choice at each level of OTree 2) 3)c1*c2*c3*c4*c5 for each [leaf,tree,sample]
        if self.bin_func=="entmoid15":                   #0.68477
            gate_values = entmoid15(threshold_logits)
        elif self.bin_func=="05_01":                     #0.67855
            gate_values = (0.5 * threshold_logits + 0.5)
            gate_values = gate_values.clamp_(-0.5, 1.5)       #(0, 1)
        elif self.bin_func == "softmax":
            gate_values = F.softmax(threshold_logits,dim=-1)
        elif self.bin_func == "05":                      #后继乏力(0.629)
            gate_values = (0.5 * threshold_logits + 0.5)
        elif self.bin_func == "":                        #后继乏力(0.630)
            gate_values = threshold_logits
        self.gate_values = gate_values
        # ^--[batch_size, num_trees, depth, 2], approximately binary
        if self.path_map is None:   #too much memory
            path_ = torch.einsum('btds,dcs->btdc', gate_values, self.bin_codes_1hot)
            # ^--[batch_size, num_trees, depth, 2 ** depth]            
        else:
            #each column is a Probability
            path_ = torch.index_select(gate_values.flatten(-2,-1),dim=-1,index=self.path_map).view(batch_size, self.num_trees,self.depth,-1)             
            assert path_.shape[-1]==2 ** self.depth

        response_weights = torch.prod(path_, dim=-2)       # ^-- [batch_size, num_trees, 2 ** depth]
        if self.config.reg_Gate!=0:
            if False:
                P = torch.sum(response_weights,dim=0)    #nTree,nLeaf
                P_all = torch.sum(P)
                alpha=torch.sum(P[:,0:nLeaf//2])/P_all
                C = -0.5*torch.log(alpha)+0.5*torch.log(1-alpha)
                #self.gates_cp = torch.sum(C)
            else:
                pass
                #self.gates_cp = torch.norm(self.gate_values,p=2)/self.gate_values.numel()
        if self.config.leaf_output == "learn_distri":
            response = torch.einsum('btl,tcl->btc', response_weights, self.response)
            # ^-- [batch_size, num_trees, distri_dim]
            return response.flatten(1, 2) if self.flatten_output else response
        elif self.config.leaf_output == "Y":        #有问题，如何validate?
            y_batch = self.config.y_batch
            leaf_value = torch.einsum('b,btl->btl', y_batch,response_weights)
            leaf_value = leaf_value.mean(dim=0)
            response = torch.einsum('btl,tl->bt', response_weights, leaf_value)
            return response
        else:
            assert(False)
            return None
        



    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        nSamp = input.shape[0]
        if  nSamp < 1000:
            print(f"!!!!!! DeTree::initialize@{self.__repr__()} has only {nSamp} sampls. This may cause instability.\n")

        with torch.no_grad():
            if self.no_attention:
                feature_values = input[:, self.feat_map]                #torch.index_select(input.flatten(), 0, self.feat_select)
                feature_values = torch.einsum('bf,f->bf', feature_values, self.feat_weight)
                feature_values=feature_values.reshape(-1,self.num_trees,self.depth)
                assert feature_values.shape[0]==input.shape[0]
            else:
                feature_values = self.get_attention_value(input)
            #feature_selectors = self.choice_function(self.feat_attention, dim=0)
            # ^--[in_features, num_trees, depth]
            #feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_trees, self.nFeature])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.nFeature)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return "{}(F={},f={} T={},D={}, response_dim={}, " \
               "init_attention={},flatten_output={},bin_func={},init_response={})".format(
            self.__class__.__name__, 0 if self.no_attention else self.feat_attention.shape[0],
            self.nFeature,self.num_trees, self.depth, self.response_dim, 
            self.init_attention_func.__name__,self.flatten_output,
            self.bin_func,
            self.init_responce_func.__name__
        )

