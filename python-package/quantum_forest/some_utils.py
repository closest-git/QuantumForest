import numpy as np
import math
import random
import torch
import sys
import os
import psutil
import glob


def split__sections(dim_0,nClass):
    split_dim = range(dim_0)
    sections=[]
    for arr in np.array_split(np.array(split_dim), nClass):
        sections.append(len(arr))
    assert len(sections) > 0
    return sections

def shrink(x0,x1,max_sz=2):
    if x1-x0>max_sz:
        center=(x1+x0)//2
        #x1 = x0+max_sz
        x1 = center + max_sz // 2
        x0 = center - max_sz // 2
    return x0,x1

def split_regions_2d(shape,nClass):
    dim_1,dim_2=shape[-1],shape[-2]
    n1 = (int)(math.sqrt(nClass))
    n2 = (int)(math.ceil(nClass/n1))
    assert n1*n2>=nClass
    section_1 = split__sections(dim_1, n1)
    section_2 = split__sections(dim_2, n2)
    regions = []
    x1,x2=0,0
    for sec_1 in section_1:
        for sec_2 in section_2:
            #box=(x1,x1+sec_1,x2,x2+sec_2)
            box = shrink(x1,x1+sec_1)+shrink(x2,x2+sec_2)
            regions.append(box)
            if len(regions)>=nClass:
                break
            x2 = x2 + sec_2
        x1 = x1 + sec_1;    x2=0
    return regions

def seed_everything(seed=0):
    print(f"======== seed_everything seed={seed}========")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #https://pytorch.org/docs/stable/notes/randomness.html

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
'''
    if fix_seed is not None:        # fix seed
        seed = fix_seed #17 * 19
        print("!!! __pyTorch FIX SEED={} use_cuda={}!!!".format(seed,use_cuda) )
        random.seed(seed-1)
        np.random.seed(seed)
        torch.manual_seed(seed+1)
        if use_cuda:
            torch.cuda.manual_seed(seed+2)
            torch.cuda.manual_seed_all(seed+3)
            torch.backends.cudnn.deterministic = True
'''

def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use in python(GB):', memoryUse)

def pytorch_env( device=0 ):
    #torch.cuda.set_device(0)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    # from subprocess import call
    # call(["nvcc", "--version"]) does not work
    # ! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    nDevive = torch.cuda.device_count()
    print ('Available devices ',nDevive )
    for i in range(nDevive):
        name = torch.cuda.get_device_name()
        print( f"****** {i}-{name} \tcapabilit={torch.cuda.get_device_capability(i)} :\n\t{torch.cuda.get_device_properties(i)}") 
    print ('Current cuda device ', torch.cuda.current_device())
    use_cuda = torch.cuda.is_available()
    print("USE CUDA=" + str(use_cuda))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    Tensor = FloatTensor
    cpuStats()
    print("===== torch_init device={}".format(device))
    return device

def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x

def OnInitInstance(seed=0):
    seed_everything(seed)
    gpu_device = pytorch_env()
    return gpu_device

def get_latest_file(pattern):
    list_of_files = glob.glob(pattern) # * means all if need specific format then *.csv
    assert len(list_of_files) > 0, "No files found: " + pattern
    return max(list_of_files, key=os.path.getctime)

def dump_model_params(model):
    nzParams = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nzParams += param.nelement()
            print(f"\t{name}={param.nelement()}")
    print(f"========All parameters={nzParams}\n\n")
    return nzParams

def model_params(model):
    nzParams = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nzParams += param.nelement()
    return nzParams

#https://stackoverflow.com/questions/1639174/creating-class-instance-properties-from-a-dictionary
class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])
        
    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        attrs = str([x for x in self.__dict__])
        return "<dict2obj: %s="">" % attrs