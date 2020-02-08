import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from .nn_utils import sparsemax, sparsemoid, ModuleWithInit,entmoid15
from .utils import check_numpy
from warnings import warn


class ODST(ModuleWithInit):
    def init_choice_weight(self,init_func,in_features, num_trees, depth,feat_info=None):
        #f = initialize_selection_logits_.__name__
        init_func = nn.init.uniform_
        self.choice_reuse = False
        self.in_features, self.num_trees, self.depth=in_features, num_trees, depth
        self.nChoice = self.num_trees*self.depth
        if self.choice_reuse:
            self.nChoice = self.nChoice//10
            print(f"======  init_choice_weight nChoice={self.nChoice}")
            self.choice_map = [random.randrange(0, self.nChoice) for _ in range(self.num_trees*self.depth)] #random.sample(range(self.nSeed), self.nChoice)
            assert len(self.choice_map)==self.num_trees*self.depth
        self.feature_selection_logits = nn.Parameter(
            torch.zeros([self.in_features, self.nChoice]), requires_grad=True
        )
        feat_val = torch.zeros([self.in_features, self.nChoice])
        init_func(feat_val)
        if feat_info is not None and 'importance' in feat_info.columns:
            importance = torch.from_numpy(feat_info['importance'].values).float()
            assert importance.shape[0]==in_features
            fmax,fmin=torch.max(importance),torch.min(importance)
            weight = importance/fmax
            #weight[weight<1.0e-4] = 1.0e-4
            weight = weight*weight*weight
            print(f"====== feat weight={weight}")
            for i in range(self.nChoice):
                #print(feat_val[:,i])
                feat_val[:,i] = feat_val[:,i]*weight
                #print(feat_val[:, i])
            self.feature_selection_logits = nn.Parameter(
                feat_val, requires_grad=True
            )
            print(f"====== init_choice_weight from feat_info={feat_info.columns}")
            pass

        self.feature_selection_logits = nn.Parameter(
            feat_val, requires_grad=True
        )
        #init_func(self.feature_selection_logits)
        print(f"====== init_choice_weight f={init_func.__name__}")

    #weights computed as entmax over the learnable feature selection matrix F ∈ R d×n
    def get_choice_weight(self):
        feature_logits = self.feature_selection_logits
        choice_weight = self.choice_function(feature_logits, dim=0)
        #feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
        # ^--[in_features, num_trees, depth]
        if self.choice_reuse:
            choice_weight = choice_weight[:, self.choice_map]
            #for i,id in enumerate(self.choice_map):
            #    self.choice[:,i] = choice_weight[:,id]
        else:
            pass
        choice_weight = choice_weight.view(self.in_features,self.num_trees,-1)

        return choice_weight

    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 choice_function=sparsemax, bin_function=sparsemoid,feat_info=None,
                 initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0,
                 ):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
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

        self.depth, self.num_trees, self.tree_dim, self.flatten_output = depth, num_trees, tree_dim, flatten_output
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.init_responce_func = initialize_response_
        self.init_choice_func = initialize_selection_logits_

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)

        self.init_choice_weight(initialize_selection_logits_,in_features, num_trees, depth,feat_info)
        '''
        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True
        )
        initialize_selection_logits_(self.feature_selection_logits)
        '''


        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_selectors = self.get_choice_weight()
        '''
        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]        
        '''

        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)
        #threshold_logits = (feature_values ) * torch.exp(-self.log_temperatures) - self.feature_thresholds

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        #RESPONSE_WEIGHTS 1)choice at each level of OTree 2) 3)c1*c2*c3*c4*c5 for each [leaf,tree,sample]
        #bins = self.bin_function(threshold_logits)
        bin_func = "05_01"
        if bin_func=="entmoid15":                   #0.68477
            bins = entmoid15(threshold_logits)
        elif bin_func=="05_01":                     #0.67855
            bins = (0.5 * threshold_logits + 0.5)
            #bins = bins.clamp_(0, 1)
            bins = bins.clamp_(-0.5, 1.5)
        elif bin_func == "05":                      #后继乏力(0.629)
            bins = (0.5 * threshold_logits + 0.5)
        elif bin_func == "":                        #后继乏力(0.630)
            bins = threshold_logits

        # ^--[batch_size, num_trees, depth, 2], approximately binary
        if True:   #too much memory
            bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
            # ^--[batch_size, num_trees, depth, 2 ** depth]
            response_weights = torch.prod(bin_matches, dim=-2)
            # ^-- [batch_size, num_trees, 2 ** depth]
        else:
            response_weights = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)

        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training. Use with torch.no_grad() for memory efficiency.")
        with torch.no_grad():
            feature_selectors = self.get_choice_weight()
            #feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return "{}(F={},T={},D={}, tree_dim={}, " \
               "flatten_output={},bin_func={},init_response={},init_choice={})".format(
            self.__class__.__name__, self.feature_selection_logits.shape[0],
            self.num_trees, self.depth, self.tree_dim, self.flatten_output,
            self.bin_function.__name__,
            self.init_responce_func.__name__,self.init_choice_func.__name__
        )

