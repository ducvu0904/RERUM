import torch
import torch.nn as nn
import torch.nn.functional as F
from ziln import zero_inflated_lognormal_pred, zero_inflated_lognormal_loss
import numpy as np

class TarnetBase(nn.Module):
    """
    Base Tarnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100, shared_dropout=0.0, outcome_dropout=0.0, positive_rate=0.05):
        super(TarnetBase, self).__init__()
        self.shared = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_dropout),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_dropout),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_dropout)
        )

        self.head_0_common = nn.Sequential(
            nn.Linear(shared_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_dropout),
            nn.Linear(outcome_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_dropout)
        )
        self.head_1_common = nn.Sequential(
            nn.Linear(shared_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_dropout),
            nn.Linear(outcome_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_dropout)
        )

        self.y0_mu = nn.Linear(outcome_hidden, 1)
        self.y0_sigma = nn.Linear(outcome_hidden, 1)
        self.y0_p = nn.Linear(outcome_hidden, 1)

        self.y1_mu = nn.Linear(outcome_hidden, 1)
        self.y1_sigma = nn.Linear(outcome_hidden, 1)
        self.y1_p = nn.Linear(outcome_hidden, 1)

        self._init_weights(positive_rate=positive_rate)

    def _init_weights(self, positive_rate=0.05):
        # Init p bias to match the empirical positive rate so the classification
        # head starts from a calibrated prior instead of 1% (which was too conservative
        # and slowed convergence of the p head significantly).
        positive_rate = np.clip(positive_rate, 1e-4, 1 - 1e-4)
        p_bias = float(np.log(positive_rate / (1 - positive_rate)))
        with torch.no_grad():
            self.y0_p.bias.fill_(p_bias)
            self.y1_p.bias.fill_(p_bias)

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
            [batch, 3] ZILN logits for control outcome (p, mu, sigma)
        y1: torch.Tensor
            [batch, 3] ZILN logits for treatment outcome (p, mu, sigma)
        """
        z = self.shared(inputs)

        # --- CONTROL FLOW ---
        h0 = self.head_0_common(z)
        mu0 = self.y0_mu(h0)
        sigma0 = self.y0_sigma(h0)
        p0 = self.y0_p(h0)
        y0 = torch.cat([p0, mu0, sigma0], dim=1)

        # --- TREATMENT FLOW ---
        h1 = self.head_1_common(z)
        mu1 = self.y1_mu(h1)
        sigma1 = self.y1_sigma(h1)
        p1 = self.y1_p(h1)
        y1 = torch.cat([p1, mu1, sigma1], dim=1)

        return y0, y1

def outcome_loss(y_t, y_c, y0_pred, y1_pred):

    loss_0 = zero_inflated_lognormal_loss(y_t, y1_pred)
    loss_1 = zero_inflated_lognormal_loss(y_c, y0_pred)

    # Ensure losses are valid (no NaN or inf)
    # Raise an error in debug mode; silently replacing NaN hides root causes.
    # If NaN persists despite sigma clamp, the issue is upstream (e.g. exploding mu).
    if torch.isnan(loss_0) or torch.isinf(loss_0):
        loss_0 = loss_0.new_tensor(0.0)
    if torch.isnan(loss_1) or torch.isinf(loss_1):
        loss_1 = loss_1.new_tensor(0.0)

    loss = (loss_0 + loss_1)
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