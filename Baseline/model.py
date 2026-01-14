import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DragonNetBase(nn.Module):
    """
    Base Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100):
        super(DragonNetBase, self).__init__()
        self.shared = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(0.2)
        )

        self.treat_out = nn.Linear(in_features=shared_hidden, out_features=1)
        
        self.y0 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=1)
        )
        
        self.y1 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=1)
        )
        
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)
        
    def forward(self, inputs):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        """

        z = self.shared(inputs)
        
        t_pred = torch.sigmoid(self.treat_out(z))
        
        y0 = self.y0(z)
        y1 = self.y1(z)
        
        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps
    


def dragonnet_loss (y_true, t_true, t_pred, y0_pred, y1_pred, alpha=1.0, ranking_lambda =1.0):
    t_pred = (t_pred + 0.01) / 1.02
    propensity_loss = torch.mean(F.binary_cross_entropy(t_pred, t_true))
    
    # print (f"losst = {propensity_loss}")
    loss0 = torch.mean((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.mean(t_true * torch.square(y_true - y1_pred))

    loss_y = loss0 + loss1  
    # print (f"loss_y = {loss_y} | uplift ranking loss = {loss_uplift_ranking} | resposne_ranking_loss = {loss_response_ranking}" )

    loss = loss0 + loss1 + alpha * propensity_loss 
    return loss

def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, beta=1.0, ranking_lambda=1.0):
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
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha, ranking_lambda)
    t_pred = (t_pred +0.01)/1.02
    
    y_pred = t_true * y1_pred + (1-t_true) * y0_pred
    
    #clever covariates
    h = (t_true / t_pred) - ((1-t_true) / (1- t_pred))
    h = torch.clamp(h, -20.0, 20.0)
    
    y_pert = y_pred + eps * h 
    targeted_regularization_raw = torch.sum((y_true - y_pert)**2)
    scale_factor = torch.mean(y_true**2).detach() + 1e-6
    normalized_tarreg_loss = targeted_regularization_raw / scale_factor
    # print (f" tarreg = {normalized_tarreg_loss}")
    return vanilla_loss + beta * normalized_tarreg_loss

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = 0
        self.best_model_state = None  # Lưu state dict của model tốt nhất

    def early_stop(self, validation_loss, epoch=None, model=None):
        # Cải thiện nếu loss giảm hơn min_validation_loss - min_delta
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch if epoch is not None else self.best_epoch
            self.counter = 0
            if model is not None:
                import copy
                self.best_model_state = copy.deepcopy(model.state_dict())
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            # Trong biên độ min_delta: không cải thiện nhưng cũng không tăng counter
            self.counter = 0
        return False

    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Restored model to best epoch {self.best_epoch} with Val Loss = {self.min_validation_loss:.4f}")
            return True
        return False