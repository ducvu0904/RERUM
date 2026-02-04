import torch
import torch.nn as nn
import numpy as np

def mmd_rbf(x, t_true, p, sigma):
    t_true = t_true.squeeze()  # Flatten to 1D for boolean indexing
    
    x_treat = x[t_true==1]
    x_control = x[t_true==0]
    
    N1 = x_treat.shape[0]
    N0 = x_control.shape[0]
    
    if N1==0 or N0==0:
        return 0
    
    tt = torch.cdist(x_treat, x_treat) **2
    cc = torch.cdist(x_control, x_control) **2
    tc = torch.cdist(x_treat, x_control) **2
    
    kernel_tt = torch.exp(-tt / (2*sigma **2))
    kernel_cc = torch.exp(-cc / (2*sigma **2))
    kernel_tc = torch.exp(-tc / (2*sigma **2))
    
    sum_ktt = (kernel_tt.sum() - N1) / (N1 *(N1-1)) 
    sum_kcc = (kernel_cc.sum() - N0) / (N0 * (N0-1))
    
    mmd = ( p** 2 * sum_ktt + (1.0 - p) ** 2 * sum_kcc - 2) - 2.0 * p * (1.0-p) * kernel_tc.mean()
    
    return 4.0 * mmd

def mmd_linear(x, t_true):
    t_true = t_true.squeeze()  # Flatten to 1D for boolean indexing
    x_treat = x[t_true==1]
    x_control = x[t_true==0]
    
    mean_treated = x_treat.mean(dim=0)
    mean_control = x_control.mean(dim=0)
    
    mmd = 2 * torch.norm(mean_treated - mean_control)
    return mmd
# 
def wasserstein(x, t_true, p=0.5, lamba = 1, iterations=10):
    t_true = t_true.squeeze()  # Flatten to 1D for boolean indexing
    x_treat = x[t_true==1]
    x_control = x[t_true==0]
    
    N1 = x_treat.shape[0]
    N0 = x_control.shape[0]
    
    if N1==0 or N0==0:  
        return 0
    
    M = torch.norm(x_treat[:, None] - x_control, dim=2)**2
    a = p * torch.ones((N1, 1)) / N1
    b = (1 - p) * torch.ones((N0, 1)) / N0
    
    k = torch.exp(-lamba * M)
    k_tilde = k/a
    u=a
    for i in range(0, iterations):
        u = 1.0 / torch.matmul(k_tilde, b/ torch.matmul(torch.transpose(k, 0, 1), u))
        
    v = b / torch.matmul(torch.transpose(k, 0, 1), u)
    T = u * (torch.transpose(v, 0, 1) * k)
    
    E = T * M
    
    return 2* torch.sum(E)