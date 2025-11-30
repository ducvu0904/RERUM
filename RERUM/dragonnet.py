import torch
import torch.nn as nn
import torch.nn.functional as F
from ziln import ZILNLoss, compute_expected_value
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
        self.headC_out = nn.Linear(outcome_hidden, 3)
        
        #---------Treatment outcome head------------
        self.headT_1 = nn.Linear(shared_hidden_layer, outcome_hidden)
        self.headT_2 = nn.Linear(outcome_hidden, outcome_hidden)
        self.headT_out = nn.Linear(outcome_hidden, 3)
        
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
        
        eps = self.epsilon(torch.ones_like(t_pred))
        
        return y0 ,y1 ,t_pred, eps

def uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
    #listwise ranking loss
    y0_pred = compute_expected_value(y0_pred)
    y1_pred = compute_expected_value(y1_pred)
    uplift_pred = y1_pred - y0_pred
    uplift_pred_t = uplift_pred[t_true==1].unsqueeze(1)
    uplift_pred_c = uplift_pred[t_true==0].unsqueeze(1)
    softmax_uplift_pred_t = F.softmax(uplift_pred_t, dim=0)
    softmax_uplift_pred_c = F.softmax(uplift_pred_c, dim=0)
    
    #ground truth
    y_t = y_true[t_true==1].unsqueeze(1)
    y_c = y_true[t_true==0].unsqueeze(1)
    
    N1 = y_t.shape[0]
    N0 = y_c.shape[0]
    
    loss = -( N1 + N0) * ((1/N1)*torch.sum(y_t * torch.log(softmax_uplift_pred_t)) - 1/N0*torch.sum(y_c*torch.log(softmax_uplift_pred_c)))
    return loss

def resposne_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
    

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
