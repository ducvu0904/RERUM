import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from ipm import mmd_linear, mmd_rbf, wasserstein

class CFRBase (nn.Module):
    def __init__(self, input_dim, shared_hidden = 200, outcome_hidden = 100, share_dropout = 0.0, outcome_dropout = 0.0):
        super(CFRBase, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(in_features= input_dim, out_features= shared_hidden),
            nn.ReLU(), 
            nn.Dropout(share_dropout),
            nn.Linear(in_features= shared_hidden, out_features= shared_hidden), 
            nn.ReLU(), 
            nn.Dropout(share_dropout),
            nn.Linear(in_features= shared_hidden, out_features= shared_hidden),   
            nn.ReLU(), 
            nn.Dropout(share_dropout)
        )
        
        self.head_1 = nn.Sequential(
            nn.Linear(in_features= shared_hidden, out_features= outcome_hidden),
            nn.ReLU(),
            nn.Dropout(outcome_dropout),
            nn.Linear(in_features= outcome_hidden, out_features= outcome_hidden),
            nn.ReLU(),
            nn.Dropout(outcome_dropout),
            nn.Linear(in_features= outcome_hidden, out_features= 1)
        )
        
        self.head_0 = nn.Sequential(
            nn.Linear(in_features= shared_hidden, out_features= outcome_hidden),
            nn.ReLU(),
            nn.Dropout(outcome_dropout),
            nn.Linear(in_features= outcome_hidden, out_features= outcome_hidden),
            nn.ReLU(),
            nn.Dropout(outcome_dropout),
            nn.Linear(in_features= outcome_hidden, out_features= 1)
        )
    
    def forward(self, input):
        z = self.shared(input)
        
        y1 = self.head_1(z)
        
        y0 = self.head_0(z)
        
        return z, y1, y0
    
def compute_ipm_loss(shared_layer, t_true, method = "mmd_rbf", alpha = 1.0):
    if alpha ==0: 
        return 0
    
    if method == 'mmd_linear':
        distance = mmd_linear(shared_layer, t_true= t_true)
    elif method == 'mmd_rbf':
        distance = mmd_rbf(shared_layer, t_true, p= 0.5, sigma=1.0)
    elif method == "wasserstein":
        distance = wasserstein(shared_layer, t_true, p=0.5, lamba = 1, iterations=10)
    else:
        distance = 0    
    return distance

def outcome_loss(y_t, y_c, y1_pred, y0_pred, ipm_loss, alpha= 1.0):
    
    y1_loss = torch.mean(torch.square(y1_pred -  y_t))
    y0_loss = torch.mean(torch.square(y0_pred -  y_c))
    
    loss = y1_loss + y0_loss + alpha * ipm_loss
    
    return loss
    
    