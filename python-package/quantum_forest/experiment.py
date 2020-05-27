import os
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from .some_utils import get_latest_file, check_numpy,model_params
#from .nn_utils import to_one_hot
from collections import OrderedDict
from copy import deepcopy
from tensorboardX import SummaryWriter

from sklearn.metrics import roc_auc_score, log_loss

def iterate_minibatch(*tensors, batch_size, shuffle=True, epochs=1,allow_incomplete=True, callback=lambda x:x):
    indices = np.arange(len(tensors[0]))
    upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """
    Computes output by applying batch-parallel function to large data tensor in chunks
    :param function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...]
    :param args: one or many tensors, each [num_instances, ...]
    :param batch_size: maximum chunk size processed in one go
    :param out: memory buffer for out, defaults to torch.zeros of appropriate size and type
    :returns: function(data), computed in a memory-efficient way
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    reg_Gate = function.L_gate.item()*batch_size 
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
        reg_Gate = reg_Gate+function.L_gate.item()*(batch_ix.stop-batch_ix.start)
    dict_info={"reg_Gate":reg_Gate/total_size,"total_size":total_size}
    return out,dict_info

class Experiment(nn.Module):
    def __init__(self, config,data,model, loss_function, experiment_name=None, warm_start=False, 
                 Optimizer=torch.optim.Adam, optimizer_params={}, verbose=False, 
                 n_last_checkpoints=1, **kwargs):
        """
        :type model: torch.nn.Module
        :param loss_function: the metric to use in trainnig
        :param experiment_name: a path where all logs and checkpoints are saved
        :param warm_start: when set to True, loads last checpoint
        :param Optimizer: function(parameters) -> optimizer
        :param verbose: when set to True, produces logging information
        """
        super().__init__()
        self.data = data
        self.model = model
        #self.loss_function = loss_function
        self.verbose = verbose
        #self.opt = Optimizer(list(self.model.parameters()), **optimizer_params)
        self.opt_parameters = optimizer_params
        self.Optimizer = Optimizer
        if hasattr(self.model,"parameters"):
            self.opt = self.Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),**self.opt_parameters )
        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints
        self.isFirstBackward = True

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}_{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)
        
        if True:
            self.experiment_path = os.path.join('logs/', experiment_name)
            if not warm_start and experiment_name != 'debug':
                if os.path.exists(self.experiment_path):        #为了批处理
                    import shutil
                    print(f'experiment {config.experiment} already exists, DELETE it!!!')
                    shutil.rmtree(self.experiment_path)
                assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        else:
            self.experiment_path = os.path.join('logs/', "cys")
        if hasattr(config,"no_inner_writer"):
            self.writer = None
        else:
            self.writer = SummaryWriter(self.experiment_path, comment=experiment_name)

        if warm_start:
            self.load_checkpoint()

        if config.feature_fraction<1:
            self.feature_fraction = config.feature_fraction
            self.select_some_feats()
        else:
            self.nFeat4Train = self.data.nFeature       
        
        
    
    def SetLearner(self,wLearner):
        self.model = wLearner        
        self.opt = self.Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),**self.opt_parameters )
        print(f"====== Experiment SetModel={wLearner}")
    
    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = "temp_{}".format(self.step)
        if path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.step)
        ]), path)
        if self.verbose:
            print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        if self.model.config.task=="predict":
            checkpoint = torch.load(path)
        else:
            assert tag is None or path is None, "please provide either tag or path or nothing, not both"
            if tag is None and path is None:
                path = get_latest_file(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
            elif tag is not None and path is None:
                path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))

        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.step = int(checkpoint['step'])

        if self.verbose:
            print('Loaded ' + path)
        return self

    def average_checkpoints(self, tags=None, paths=None, out_tag='avg', out_path=None):
        assert tags is None or paths is None, "please provide either tags or paths or nothing, not both"
        assert out_tag is not None or out_path is not None, "please provide either out_tag or out_path or both, not nothing"
        if tags is None and paths is None:
            paths = self.get_latest_checkpoints(
                os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'), self.n_last_checkpoints)
        elif tags is not None and paths is None:
            paths = [os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(tag)) for tag in tags]

        checkpoints = [torch.load(path) for path in paths]
        averaged_ckpt = deepcopy(checkpoints[0])
        for key in averaged_ckpt['model']:
            values = [ckpt['model'][key] for ckpt in checkpoints]
            averaged_ckpt['model'][key] = sum(values) / len(values)

        if out_path is None:
            out_path = os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(out_tag))
        torch.save(averaged_ckpt, out_path)

    def get_latest_checkpoints(self, pattern, n_last=None):
        list_of_files = glob.glob(pattern)
        assert len(list_of_files) > 0, "No files found: " + pattern
        return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

    def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
        if number_ckpts_to_keep is None:
            number_ckpts_to_keep = self.n_last_checkpoints
        paths = self.get_latest_checkpoints(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        paths_to_delete = paths[number_ckpts_to_keep:]

        for ckpt in paths_to_delete:
            os.remove(ckpt)
    
    def select_some_feats(self,isDump=True):
        nPickFeat = (int)(self.data.nFeature*self.feature_fraction)
        self.feat_cands = random.choices(population = list(range(self.data.nFeature)),k = nPickFeat)
        #self.feat_cands = list(range(self.data.nFeature))
        self.nFeat4Train = len(self.feat_cands)
        if isDump:
            print(f"====== select_some_feats len={self.nFeat4Train} feat=[{self.feat_cands[0:5]}...{self.feat_cands[-5:-1]}]")

    def loss_on_batch(self,y_output,y_batch):
        #loss = self.loss_function(y_output, y_batch).mean() 
        if self.data.problem()=="classification":
            #assert list(y_output.shape)==[y_batch.shape,self.data.nClasses]
            loss = F.cross_entropy(y_output,y_batch.long())
        else:
            assert y_output.shape==y_batch.shape
            loss = F.mse_loss(y_output, y_batch.float())
        loss = loss.mean()
        
        #loss = self.model.reg_L1*self.model.config.reg_L1
        loss = loss+self.model.reg_L1*self.model.config.reg_L1+self.model.L_gate*self.model.config.reg_Gate 
        #print(f"\t{torch.min(loss)}:{torch.max(loss)}")
        return loss

    def train_on_batch(self, *batch, device):
        #with torch.autograd.set_detect_anomaly(True):
        x_batch, y_batch = batch
        augment = self.model.config.Augment
        if augment["batch_noise"]>0:
            noise = augment["batch_noise"]
            stds = np.std(x_batch, axis=0, keepdims=True)
            #noise_std = noise / np.maximum(stds, noise)
            noise = noise*stds * np.random.randn(*x_batch.shape)
            x_aug = x_batch+noise
            x_batch = torch.as_tensor(x_aug, device=device, dtype=torch.float32)
        else:
            x_batch = torch.as_tensor(x_batch, device=device)
        y_batch = torch.as_tensor(y_batch, device=device)
        self.y_batch = y_batch
        self.model.train()
        self.opt.zero_grad()
        y_output=self.model(x_batch)

        loss = self.loss_on_batch(y_output,y_batch)
        if False:
            if self.data.problem()=="classification":
                #assert list(y_output.shape)==[y_batch.shape,self.data.nClasses]
                loss = F.cross_entropy(y_output,y_batch.long())
            else:
                assert y_output.shape==y_batch.shape
                loss = F.mse_loss(y_output, y_batch)
            loss = loss.mean()
            #loss = self.model.reg_L1*self.model.config.reg_L1
            loss = loss+self.model.reg_L1*self.model.config.reg_L1+self.model.L_gate*self.model.config.reg_Gate 
        
        loss.backward()     #retain_graph=self.isFirstBackward        
        self.isFirstBackward = False
        self.opt.step()
        self.step += 1
        self.y_batch = None
        if self.writer is not None:     self.writer.add_scalar('train loss', loss.item(), self.step)
        self.metrics = {'loss': loss.item()}
        del loss       
        #if self.feature_fraction<1:self.select_some_feats(isDump=False)
        return self.metrics

    def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits,dict_info = process_in_chunks(self.model, X_test, batch_size=batch_size)
            logits = check_numpy(logits)
            error_rate = (y_test != np.argmax(logits, axis=1)).mean()
        dict_info["ERR"] = error_rate
        #return error_rate
        return dict_info

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction,dict_info = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean()    
        if torch.cuda.is_available():   
            torch.cuda.empty_cache()
        dict_info["ERR"] = error_rate
        return dict_info,prediction
    
    def evaluate_auc(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            auc = roc_auc_score(check_numpy(to_one_hot(y_test)), logits)
        return auc
    
    def evaluate_logloss(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.tensor(y_test)
            logloss = log_loss(check_numpy(to_one_hot(y_test)), logits)
        return logloss
    
    def AfterEpoch(self,epoch,XX_,YY_,best_mse,isTest=False,isPredict=False):
        t0=time.time()
        config = self.model.config
        self.y_batch = None
        if isTest:
            self.load_checkpoint(tag='best_mse')
        elif config.average_training: #有意思，最终可以提高Val MSE
            self.save_checkpoint()
            self.average_checkpoints(out_tag='avg')
            self.load_checkpoint(tag='avg')

        if self.data.problem()=="classification":
            dict_info = self.evaluate_classification_error(XX_,YY_, device=config.device, batch_size=config.eval_batch_size)
            mse = dict_info['ERR']
        else:
            dict_info,prediction = self.evaluate_mse(XX_,YY_, device=config.device, batch_size=config.eval_batch_size)
            prediction = self.data.Y_trans(prediction)
            if config.err_relative:
                nz = len(YY_)
                lenY = np.linalg.norm(YY_,2) 
                lenOff = np.linalg.norm(YY_ - prediction,2)
                avg_Y2 = ((YY_) ** 2).mean()
                #d_1 = lenY*lenY-nrm_Y*nz
                #d_2 = lenOff*lenOff-((YY_ - prediction) ** 2).mean()*nz                 
                mse_0 = ((YY_ - prediction) ** 2).mean()/avg_Y2
                mse = lenOff/lenY       #relative error
                d_3 = mse*mse-mse_0
            else:
                mse = ((YY_ - prediction) ** 2).mean()    #显然相对误差更合理欸
                #rmse = sqrt(mse)  
        
        if config.cascade_LR:
            prediction=LinearRgressor.AfterPredict(XX_,prediction)   


        self.model.AfterEpoch(isBetter=mse < best_mse, accu=mse,epoch=epoch)
        reg_Gate = dict_info["reg_Gate"] 
        loss_step = self.metrics['loss']
        if isTest:
            print(f'====== Best step: {self.step} test={XX_.shape} ACCU@Test={mse:.5f} \treg_Gate:{reg_Gate:.4g}time={time.time()-t0:.2f}' )
        else:
            print(  f"\n{self.step}\tnP={model_params(self.model)},nF4={self.nFeat4Train}\t{loss_step:.5f}\treg_Gate:{reg_Gate:.4g}\tT={time.time()-t0:.2f}"\
                f"\tVal MSE:{mse:.4f}" )  


        if False:   #two stage training 真令人失望啊
            all_grad = False if epoch%2==1 else True
            self.model.freeze_some_params({"requires_grad":all_grad})     
            nzParams = model_params(self.model)           
            print(f"\tEpoch={epoch}\tSet all_grad to {all_grad}  nzParams={nzParams} !!!")
            self.opt = self.Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), **self.opt_parameters)
        return mse
