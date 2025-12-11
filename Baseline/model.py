import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DragonNetBase(nn.Module):
    """
    Base Dragonnet model
    
    parameter
    ----------
    input_dim: int
        input dimension for covariates
    shared_hidden_layer: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    
    def __init__(self, input_dim, shared_hidden_layer=200, outcome_hidden =100):
        super(DragonNetBase, self).__init__()
        
        #---------Shared layers----------
        self.full_connect_1 = nn.Linear(input_dim, shared_hidden_layer)
        self.full_connect_2 = nn.Linear(shared_hidden_layer, shared_hidden_layer)
        self.full_connect_3 = nn.Linear(shared_hidden_layer, shared_hidden_layer)
        
        #---------Control outcome head-------------
        self.headC_1 = nn.Linear(shared_hidden_layer, outcome_hidden)
        self.headC_2 = nn.Linear(outcome_hidden, outcome_hidden)
        self.headC_out = nn.Linear(outcome_hidden, 1)
        
        #---------Treatment outcome head------------
        self.headT_1 = nn.Linear(shared_hidden_layer, outcome_hidden)
        self.headT_2 = nn.Linear(outcome_hidden, outcome_hidden)
        self.headT_out = nn.Linear(outcome_hidden, 1)
        
        #---------Propensity score-----------------
        self.propensity_head = nn.Linear(shared_hidden_layer,1)
        
        self.epsilon = nn.Linear(1,1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)
        
    def forward(self, inputs):
        """
        Parameters
        ------------
        inputs: torch.sensor
                covariates
            
        Returns
        ------------
        y0: torch.sensor
            outcome under control
        y1: torch.sensor
            outcome under treatment 
        t_pred: torch.sensor
            predicited treatment
        """
        #shared layer
        x = F.relu(self.full_connect_1(inputs))
        x = F.relu(self.full_connect_2(x))
        z = F.relu(self.full_connect_3(x))
        
        #propensity 
        t_pred = torch.sigmoid(self.propensity_head(z))
        
        #outcome control
        y0 = F.relu(self.headC_1(z))
        y0 = F.relu(self.headC_2(y0))
        y0 = F.relu(self.headC_out(y0))

        #outcome treatment
        y1 = F.relu(self.headT_1(z))
        y1 = F.relu(self.headT_2(y1))
        y1 = F.relu(self.headT_out(y1))
        
        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])
        
        return y0 ,y1 ,t_pred, eps

def dragonnet_loss (y_true, t_true, t_pred, y0_pred, y1_pred, alpha=1.0):
    t_pred = (t_pred + 0.01) / 1.02
    propensity_loss = torch.sum(F.binary_cross_entropy(t_pred, t_true))
    

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    
    loss = loss0 + loss1 + alpha * propensity_loss
    return loss

def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, beta=1.0):
    """
    Parameters
    ---------------
    y_true: true spending
    t_true: true treatment
    t_pred: predicted treatment
    y0_pred: predicted revenue under control
    y1_pred: predicted revenue under treatment
    eps: trainable epsilon parameter

    Returns
    ---------------
    loss
    """
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha)
    t_pred = (t_pred +0.01)/1.02
    
    y_pred = t_true * y1_pred + (1-t_true) * y0_pred
    
    #clever covariates
    h = (t_true / t_pred) - ((1-t_true) / (1- t_pred))
    y_pert = y_pred + eps * h 
    targeted_reg = torch.sum((y_true - y_pert)**2)
    
    return vanilla_loss + beta * targeted_reg

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False