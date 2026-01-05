import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TarnetBase(nn.Module):
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
        super(TarnetBase, self).__init__()
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

        self.y0 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=outcome_hidden, out_features=1),
        )
        
        
        self.y1 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=outcome_hidden, out_features=1)
        )
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
        y0 = self.y0(z)
        y1 = self.y1(z)
        return y0, y1

def tarnet_loss(y_true, t_true, y0_pred, y1_pred):
    
    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    
    loss = loss0 + loss1
    return loss
        
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