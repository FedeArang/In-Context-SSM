import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Users/faran/Documents_nuova/Documenti/Federico/ETH/in context learning/Git_Hub/In-Context-SSM/model')
from model.hippo import HiPPO_LegT
from model.hippo import HiPPO_LegS
from model.hippo import HiPPO_FouT


def plot(test_data, config, return_losses=False):
    
    N = config['N']
    T = config['T']
    filename = config['filename']

    if config['experiment']=='function_approximation':
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
        it = iter(test_loader)
        f, _ = next(it)
        f, _ = next(it)
        f = f.squeeze(0).squeeze(-1)

    else:
        f = test_data

    legt = HiPPO_LegT(N, 1./T)
    #f_legt = legt.reconstruct(legt(f))[-1]
    f_legt = legt(f)
    f_legt = torch.reshape(f_legt, (T, ))

    legs = HiPPO_LegS(N, T)
    f_legs = legs(f)
    f_legs = torch.reshape(f_legs, (T, ))

    fout = HiPPO_FouT(N, 1./T)
    f_fout = fout(f)
    f_fout = torch.reshape(f_fout, (T, ))

    #print(F.mse_loss(f[1::], f_legt[0:-1]))
    #print(F.mse_loss(f[1::], f_legs[0:-1]))
    #print(F.mse_loss(f[1::], f_fout[0:-1]))

    vals = np.linspace(0.0, 1.0, T)
    plt.figure(figsize=(6, 2))
    plt.plot(vals[1::], f[1::]+0.1, 'k', linewidth=1.0)
    plt.plot(vals[1:T//1], f_legt[1:T//1])
    plt.plot(vals[:T//1], f_legs[:T//1])
    plt.plot(vals[:T//1], f_fout[:T//1])
    plt.xlabel('Time (normalized)', labelpad=-10)
    plt.xticks([0, 1])
    plt.legend(['f', 'legt', 'legs', 'fout'])
    plt.savefig(f'{filename}', bbox_inches='tight')
    # plt.show()
    plt.close()

    #plt.plot(vals[1::], torch.abs(f[1::]-f_legt[0:-1]))
    #plt.show()

    if return_losses:
        return F.mse_loss(f[1::], f_legt[0:-1]), F.mse_loss(f[1::], f_legs[0:-1]), F.mse_loss(f[1::], f_fout[0:-1])