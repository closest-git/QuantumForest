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

        if data_set=="YEAR":
            self.depth, self.batch_size, self.nTree = 5, 1024, 256  # 0.6355-0.6485(choice_reuse)
            self.depth, self.batch_size, self.nTree = 5, 256, 2048  # 0.619
            # depth, batch_size, nTree = 7, 256, 512             #区别不大，而且有显存泄漏
        elif data_set=="YAHOO":
            self.depth, self.batch_size, self.nTree = 5, 256, 2048  # 0.619

    def model_info(self):
        return "QF_shallow"

    def env_title(self):
        title=f"{self.support.value}"
        if self.isFC:       title += "[FC]"
        if self.custom_legend is not None:
            title = title + f"_{self.custom_legend}"
        return title

    def __repr__(self):
        main_str = f"{self.data_set}_ depth={self.depth} batch={self.batch_size} nTree={self.nTree} " \
            f"choice=[{self.choice_func}] feat_info={self.feat_info}"
        #if self.isFC:       main_str+=" [FC]"
        if self.custom_legend is not None:
            main_str = main_str + f"_{self.custom_legend}"
        return main_str