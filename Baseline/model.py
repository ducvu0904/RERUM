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
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100, outcome_dropout=0.2):
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
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=1)
        )
        
        self.y1 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(outcome_dropout),
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
    


def dragonnet_loss(
    y_t, y_c,                    # Ground truth đã split
    y0_pred_t, y0_pred_c,        # Y0 predictions đã split  
    y1_pred_t, y1_pred_c,        # Y1 predictions đã split
    t_pred, t_true, eps,         # Propensity và treatment labels (full batch)
    alpha=1.0, 
    response_lambda=1.0, 
    uplift_lambda=1.0,
    max_samples=200
):
    """
    DragonNet loss với dual stream input.
    
    Args:
        y_t, y_c: Ground truth của treatment và control groups
        y0_pred_t, y0_pred_c: Y0 predictions cho treatment và control
        y1_pred_t, y1_pred_c: Y1 predictions cho treatment và control  
        t_pred: Propensity predictions (full batch)
        t_true: Treatment labels (full batch)
        eps: Epsilon từ model
        alpha: Weight cho propensity loss
        response_lambda: Weight cho response ranking loss
        uplift_lambda: Weight cho uplift ranking loss
        max_samples: S trong paper - số samples để tính ranking loss
    """
    # Propensity loss (trên full batch)
    t_pred_clipped = torch.clamp(t_pred, 0.01, 0.99)
    loss_t = torch.mean(F.binary_cross_entropy(t_pred_clipped, t_true))    
    # ZILN
    loss1 = torch.mean((y_t  - y1_pred_t)**2)
    loss0 = torch.mean((y_c - y0_pred_c)**2)
    loss_y = (loss0 + loss1)/2
    

    loss = loss_y + alpha * loss_t 
    
    return loss
    
def tarreg_loss(
    y_t, y_c,                    # Ground truth đã split
    y0_pred_t, y0_pred_c,        # Y0 predictions đã split
    y1_pred_t, y1_pred_c,        # Y1 predictions đã split
    t_pred, t_true, eps,         # Propensity và treatment labels (full batch)
    n_t,                         # Số samples trong treatment batch (để split lại eps, t_pred)
    alpha=1.0, 
    beta=1.0, 
    response_lambda=1.0, 
    uplift_lambda=1.0,
    max_samples=200
):
    """
    Targeted regularization loss với dual stream input.
    
    Args:
        y_t, y_c: Ground truth của treatment và control groups
        y0_pred_t, y0_pred_c: Y0 predictions cho treatment và control
        y1_pred_t, y1_pred_c: Y1 predictions cho treatment và control
        t_pred: Propensity predictions (full batch)
        t_true: Treatment labels (full batch)  
        eps: Epsilon từ model (full batch)
        n_t: Số samples trong treatment batch
        alpha: Weight cho propensity loss
        beta: Weight cho targeted regularization
        response_lambda: Weight cho response ranking loss
        uplift_lambda: Weight cho uplift ranking loss
        max_samples: S trong paper
    """
    # DragonNet loss
    vanilla_loss = dragonnet_loss(
        y_t, y_c, 
        y0_pred_t, y0_pred_c, 
        y1_pred_t, y1_pred_c,
        t_pred, t_true, eps,
        alpha=alpha, 
        response_lambda=response_lambda, 
        uplift_lambda=uplift_lambda,
        max_samples=max_samples
    )
    
    # Targeted regularization
    t_pred_clipped = (t_pred + 0.01) / 1.02
    
    # Split predictions theo t/c để tính y_pred
    t_pred_t = t_pred_clipped[:n_t]
    t_pred_c = t_pred_clipped[n_t:]
    t_true_t = t_true[:n_t]  # Sẽ là 1s
    t_true_c = t_true[n_t:]  # Sẽ là 0s
    eps_t = eps[:n_t]
    eps_c = eps[n_t:]
    
    # Convert ZILN predictions
    y0_pred_exp_t = zero_inflated_lognormal_pred(y0_pred_t)
    y0_pred_exp_c = zero_inflated_lognormal_pred(y0_pred_c)
    y1_pred_exp_t = zero_inflated_lognormal_pred(y1_pred_t)
    y1_pred_exp_c = zero_inflated_lognormal_pred(y1_pred_c)
    
    # y_pred = t * y1 + (1-t) * y0
    y_pred_t = t_true_t * y1_pred_exp_t + (1 - t_true_t) * y0_pred_exp_t  # = y1_pred_exp_t (vì t=1)
    y_pred_c = t_true_c * y1_pred_exp_c + (1 - t_true_c) * y0_pred_exp_c  # = y0_pred_exp_c (vì t=0)
    
    # h = t/e - (1-t)/(1-e)
    h_t = (t_true_t / t_pred_t) - ((1 - t_true_t) / (1 - t_pred_t))  # = 1/e (vì t=1)
    h_c = (t_true_c / t_pred_c) - ((1 - t_true_c) / (1 - t_pred_c))  # = -1/(1-e) (vì t=0)
    
    # y_pert = y_pred + eps * h
    y_pert_t = y_pred_t + eps_t * h_t
    y_pert_c = y_pred_c + eps_c * h_c
    
    # Targeted regularization trên cả 2 groups
    tarreg_t = torch.sum((y_t - y_pert_t)**2)
    tarreg_c = torch.sum((y_c - y_pert_c)**2)
    targeted_regularization_raw = (tarreg_t + tarreg_c)
    
    # Normalize
    # y_all = torch.cat([y_t, y_c], dim=0)
    # scale_factor = torch.mean(y_all**2).detach() + 1e-6
    # normalized_tarreg_loss = beta * (targeted_regularization_raw / scale_factor)
    
    loss = vanilla_loss + beta * targeted_regularization_raw
    
    return loss
    

        
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


class QiniEarlyStopper:
    """Early stopper for maximizing Qini coefficient with model checkpoint"""
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_qini = -np.inf
        self.best_epoch = 0
        self.best_model_state = None  # Lưu state dict của model tốt nhất

    def early_stop(self, qini_score, epoch, model=None):
        """
        Check early stopping condition và lưu model state nếu có cải thiện.
        
        Parameters
        ----------
        qini_score: float
            Qini coefficient của epoch hiện tại
        epoch: int
            Epoch hiện tại
        model: nn.Module, optional
            Model để lưu state dict khi có kết quả tốt nhất
            
        Returns
        -------
        bool: True nếu nên dừng training
        """
        if qini_score > (self.best_qini + self.min_delta):
            self.best_qini = qini_score
            self.best_epoch = epoch
            self.counter = 0
            # Lưu model state khi có kết quả tốt nhất
            if model is not None:
                import copy
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def restore_best_model(self, model):
        """
        Khôi phục model về trạng thái tốt nhất.
        
        Parameters
        ----------
        model: nn.Module
            Model cần khôi phục
            
        Returns
        -------
        bool: True nếu khôi phục thành công, False nếu không có state để khôi phục
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Restored model to best epoch {self.best_epoch} with Qini = {self.best_qini:.4f}")
            return True
        return False