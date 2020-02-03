import os, sys
import time
sys.path.insert(0, './python-package/')
import numpy as np
import matplotlib.pyplot as plt
import node_lib
import torch, torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment_name = 'year_node_shallow'
experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}_{:0>2d}'.format(experiment_name, *time.gmtime()[:5])
print("experiment:", experiment_name)
data = node_lib.Dataset("YEAR", random_state=1337, quantile_transform=True, quantile_noise=1e-3)
in_features = data.X_train.shape[1]

mu, std = data.y_train.mean(), data.y_train.std()
normalize = lambda x: ((x - mu) / std).astype(np.float32)
data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

print("mean = %.5f, std = %.5f" % (mu, std))

model = nn.Sequential(
    node_lib.DenseBlock(in_features, 2048, num_layers=1, tree_dim=3, depth=6, flatten_output=False,
                   choice_function=node_lib.entmax15, bin_function=node_lib.entmoid15),
    node_lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree

).to(device)

def dump_model_params(model):
    nzParams = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nzParams += param.nelement()
            print(f"\t{name}={param.nelement()}")
    print(f"========All parameters={nzParams}")
    return nzParams
print(model)
dump_model_params(model)

with torch.no_grad():
    res = model(torch.as_tensor(data.X_train[:1000], device=device))
    # trigger data-aware init

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

from qhoptim.pyt import QHAdam
optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }
trainer = node_lib.Trainer(
    model=model, loss_function=F.mse_loss,
    experiment_name=experiment_name,
    warm_start=False,
    Optimizer=QHAdam,
    optimizer_params=optimizer_params,
    verbose=True,
    n_last_checkpoints=5
)
from tqdm import tqdm
from IPython.display import clear_output
loss_history, mse_history = [], []
best_mse = float('inf')
best_step_mse = 0
early_stopping_rounds = 5000
report_frequency = 100
#batch_size=512     batch_size=1024
print(f"trainer.model={trainer.model}\ntrainer.loss_function={trainer.loss_function}\ntrainer.opt={trainer.opt}")
for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, batch_size=128,
                                     shuffle=True, epochs=float('inf')):
    metrics = trainer.train_on_batch(*batch, device=device)

    loss_history.append(metrics['loss'])

    if trainer.step % report_frequency == 0:
        trainer.save_checkpoint()
        trainer.average_checkpoints(out_tag='avg')
        trainer.load_checkpoint(tag='avg')
        mse = trainer.evaluate_mse(
            data.X_valid, data.y_valid, device=device, batch_size=1024)

        if mse < best_mse:
            best_mse = mse
            best_step_mse = trainer.step
            trainer.save_checkpoint(tag='best_mse')
        mse_history.append(mse)

        trainer.load_checkpoint()  # last
        trainer.remove_old_temp_checkpoints()

        clear_output(True)
        plt.figure(figsize=[18, 6])
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.title('Loss')
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(mse_history)
        plt.title('MSE')
        plt.grid()
        plt.show()
        print("Loss %.5f" % (metrics['loss']))
        print("Val MSE: %0.5f" % (mse))
    if trainer.step > best_step_mse + early_stopping_rounds:
        print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
        print("Best step: ", best_step_mse)
        print("Best Val MSE: %0.5f" % (best_mse))
        break

trainer.load_checkpoint(tag='best_mse')
mse = trainer.evaluate_mse(data.X_test, data.y_test, device=device)
print('Best step: ', trainer.step)
print("Test MSE: %0.5f" % (mse))