class QForest_config:
    def __init__(self,data_set, lr_base, nLayer=1,
                 choice_func="r_0.5",feat_info = None,random_seed=42,
                 ):
        self.model = "QForest"
        #self.tree_type = tree_type
        self.data_set = data_set
        self.lr_base = lr_base
        self.nLayer = nLayer
        self.seed = random_seed
        #seed_everything(self.seed)
        #self.init_value = init_value  # "random"  "zero"
        self.choice_func = choice_func
        self.rDrop = 0
        self.custom_legend = None
        self.feat_info = feat_info
        self.no_attention = False
        self.max_features = None
        self.input_dropout = 0.0
        self.num_layers = 1
        self.flatten_output = True

        if data_set=="YEAR":
            self.depth, self.batch_size, self.nTree = 5, 1024, 256  # 0.6355-0.6485(choice_reuse)
            self.depth, self.batch_size, self.nTree = 5, 256, 2048  # 0.619
            # depth, batch_size, nTree = 7, 256, 512             #区别不大，而且有显存泄漏
        elif data_set=="YAHOO":
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 256, 2048,5
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 512, 2048, 3
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 1024, 1024, 3  # 0.5943 收敛快
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 256, 4096, 3  # 0.5895
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 256, 6144, 3  # 0.5895
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 256, 2048, 3  # 0.5913
            #tree_dim=  5-0.5910;  3-0.5913
        elif data_set=="MICROSOFT":
            self.depth, self.batch_size, self.nTree, self.tree_dim = 5, 256, 2048,3

    def model_info(self):
        return "QF_shallow"

    def env_title(self):
        title=f"{self.support.value}"
        if self.isFC:       title += "[FC]"
        if self.custom_legend is not None:
            title = title + f"_{self.custom_legend}"
        return title

    def __repr__(self):
        main_str = f"{self.data_set}_ depth={self.depth} batch={self.batch_size} nTree={self.nTree} tree_dim={self.tree_dim} " \
            f"NO_ATTENTION={self.no_attention} choice=[{self.choice_func}] feat_info={self.feat_info}"
        #if self.isFC:       main_str+=" [FC]"
        if self.custom_legend is not None:
            main_str = main_str + f"_{self.custom_legend}"
        return main_str