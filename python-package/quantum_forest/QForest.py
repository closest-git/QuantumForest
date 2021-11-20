
'''
@Author: Yingshi Chen

@Date: 2020-02-14 11:59:10
@
# Description: 
'''
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class DeepLogger(SummaryWriter):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = -1
        self.batch_idx = -1

        print(f"DeepLogger@{self.log_dir}......")

    def isPlot(self):
        return False
        #return self.batch_idx==2

class QForest_config:
    def __init__(self,data, lr_base, nLayer=1,choice_func="r_0.5",feat_info = None,random_seed=42):
        self.model = "QForest"
        #self.tree_type = tree_type
        self.data = data
        self.data_set = data.name;      data_set = data.name
        self.lr_base = lr_base
        self.nLayer = nLayer
        self.seed = random_seed
        #seed_everything(self.seed)
        #self.init_value = init_value  # "random"  "zero"
        self.choice_func = choice_func
        self.rDrop = 0
        self.custom_legend = None
        self.feat_info = feat_info
        # self.no_attention = False       #always False since 5/28/2020
        self.max_features = None
        self.input_dropout = 0        #YAHOO 0.59253-0.59136 没啥用
        self.num_layers = 1
        self.flatten_output = True
        self.max_out = True
        self.plot_root = "./results/"
        self.plot_train = False
        self.plot_attention = True
        self.data_normal = ""       #"NN"     "Quantile"   "BN" (0.589-0.599) BN确实差很多，奇怪
        self.leaf_output = "leaf_distri"       #"distri2fc" "distri2CNN"  "Y" "leaf_distri"
        self.reg_L1 = 0.01        #-4,-5,-6,-7,-8  -7略有提高
        self.reg_Gate = 0.1
        self.path_way="TREE_map"   #"TREE_map",   "TREE_map",   "OBLIVIOUS_map","OBLIVIOUS_1hot"
        self.support_vector = "1"   #"0"    "1"    "2"        
        
        self.back_bone = 'resnet18_x'
        self.Augment = {"batch_noise":0}    #0.0001
        self.QF_fit = 1
        self.feature_fraction = 1   #0.7
        self.bagging_fraction = 1   #0.7
        self.trainer = None;     self.cascade_LR = False;     self.average_training = False   #True不合适，难以和其它模块一起训练

        self.err_relative = False
        self.task = "train"
        self.attention_alg = "eca_response"       # 'eca_input'

        self.nMostEpochs = 1000
        self.depth, self.batch_size, self.nTree, self.response_dim, self.nLayers = 5, 256, 256, 8, 1

        if data_set=="YEAR":
            self.feat_info = None
            self.depth, self.batch_size, self.nTree = 5, 1024, 256  # 0.6355-0.6485(choice_reuse)
            self.depth, self.batch_size, self.nTree = 5, 256, 2048  # 0.619
            # depth, batch_size, nTree = 7, 256, 512             #区别不大，而且有显存泄漏
            self.depth, self.batch_size, self.nTree, self.response_dim, self.nLayers = 4, 256, 256, 3, 1
            if self.leaf_output == "distri2CNN":  
                self.depth = 4;     self.batch_size=256;       self.data_normal = ""
                self.nLayers=1;     self.response_dim = 3;      
                self.T_w = 16;      self.T_h = 16;              self.nTree = self.T_w*self.T_h; 
                self.lr_base = self.lr_base/2
        elif data_set=="YAHOO":
            #反复测试 self.response_dim=5要优于3
            self.depth, self.batch_size, self.nTree, self.response_dim = 5, 256, 2048, 3  # 0.5913->0.5892(maxout)
            self.depth, self.batch_size, self.nTree, self.response_dim, self.nLayers = 5, 256, 2048, 3, 1  #
            self.depth, self.batch_size, self.nTree, self.response_dim, self.nLayers = 4, 256, 1024, 3, 1  #
            #nLayers 4-0.58854  3-0.58982   2-0.58769
            #response_dim=  5-0.5910;  3-0.5913
            if self.leaf_output == "distri2CNN":  
                self.depth = 5;     self.batch_size=256;       self.data_normal = ""
                self.nLayers=1;     self.response_dim = 3;      
                self.T_w = 16;      self.T_h = 16;              self.nTree = self.T_w*self.T_h; 
                self.lr_base = self.lr_base/2
        elif data_set=="MICROSOFT":
            self.leaf_output = "leaf_distri"
            self.depth, self.batch_size, self.nTree, self.response_dim, self.nLayers = 5, 256, 256, 8, 1
            if self.leaf_output == "distri2CNN":  
                self.depth = 5;     self.batch_size=256;       self.data_normal = ""
                self.nLayers=1;     self.response_dim = 8;      
                self.T_w = 16;      self.T_h = 16;              self.nTree = self.T_w*self.T_h; 
                self.lr_base = self.lr_base/2
        elif data_set=="CLICK":
            self.response_dim = data.nClasses
            self.depth, self.batch_size, self.nTree, self.nLayers = 5, 256, 256, 1
            if self.leaf_output == "distri2CNN":  
                self.depth = 5;     self.batch_size=256;       self.data_normal = ""
                self.nLayers=1;        
                self.T_w = 16;      self.T_h = 16;              self.nTree = self.T_w*self.T_h; 
                self.lr_base = self.lr_base/2
        elif data_set=="HIGGS":
            self.response_dim = data.nClasses
            self.depth, self.batch_size, self.nTree, self.nLayers = 5, 1024, 256, 1
            if self.leaf_output == "distri2CNN":  
                self.depth = 5;     self.batch_size=512;       self.data_normal = ""
                self.nLayers=1;        
                self.T_w = 16;      self.T_h = 16;              self.nTree = self.T_w*self.T_h; 
                self.lr_base = self.lr_base/2
        
        if self.data_normal == "NN":
            self.feat_info = ""

        self.InitLogWriter(root_path="F:/papers/",datas_name=data_set)
    
    def InitLogWriter(self,root_path,datas_name,sFac="",sGPU=""):
        self.verbose = True
                
        log_dir = os.path.join(f"{root_path}/logs/", "{}/__{}_{}_{}_{}".format(datas_name,self.model, sFac, sGPU,datetime.datetime.now().strftime('%m-%d_%H-%M')))
        os.makedirs(log_dir, exist_ok=True)
        self.log_writer = DeepLogger(log_dir) if self.verbose else None

    def problem(self):
        return self.data.problem()
        
    def model_info(self):
        if self.model == "GBDT":
            return "GBDT"
        else:
            return "QF_shallow"

    def env_title(self):
        title=f"{self.support.value}"
        if self.isFC:       title += "[FC]"
        if self.custom_legend is not None:
            title = title + f"_{self.custom_legend}"
        return title

    def main_str(self):
        main_str = f"{self.data_set}_ layers={self.nLayer} depth={self.depth} batch={self.batch_size} nTree={self.nTree} response_dim={self.response_dim} " \
            f"\nmax_out={self.max_out} choice=[{self.choice_func}] feat_info={self.feat_info}" \
            f"\nATTENTION={self.config.attention_alg} reg_L1={self.reg_L1} path_way={self.path_way}"
        #if self.isFC:       main_str+=" [FC]"
        if self.custom_legend is not None:
            main_str = main_str + f"_{self.custom_legend}"
        return main_str
