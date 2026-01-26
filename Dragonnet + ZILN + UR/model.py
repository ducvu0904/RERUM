import torch
import torch.nn as nn
import torch.nn.functional as F
from ziln import zero_inflated_lognormal_pred, zero_inflated_lognormal_loss
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
                    
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100, shared_drop = 0.0, outcome_drop =0.0):
        super(DragonNetBase, self).__init__()
        self.shared = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_drop),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_drop),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_drop)
        )

        self.treat_out = nn.Linear(in_features=shared_hidden, out_features=1)
        
        self.head_0_common = nn.Sequential(
            nn.Linear(shared_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_drop),
            nn.Linear(outcome_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_drop)
        )
        self.head_1_common = nn.Sequential(
            nn.Linear(shared_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_drop),
            nn.Linear(outcome_hidden, outcome_hidden), nn.ReLU(), nn.Dropout(outcome_drop)
        )
        
        self.y0_mu = nn.Linear(outcome_hidden, 1)    # Chuyên trị Mu
        self.y0_sigma = nn.Linear(outcome_hidden, 1) # Chuyên trị Sigma
        self.y0_p = nn.Linear(outcome_hidden, 1)     # Chuyên trị P

        # Treatment Heads
        self.y1_mu = nn.Linear(outcome_hidden, 1)
        self.y1_sigma = nn.Linear(outcome_hidden, 1)
        self.y1_p = nn.Linear(outcome_hidden, 1)
        
        # self._init_weights()
        
        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)
        
    # def _init_weights(self):
    #     # 1. Khởi tạo chung cho toàn bộ module (Xavier)
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    #     # 2. Khởi tạo BIAS riêng cho các nhánh (Quan trọng)
    #     # Thay vì dùng _init_head_bias cũ, ta set trực tiếp
        
    #     # --- CONTROL HEADS ---
    #     # Set P bias = -4.6 (để xác suất ban đầu ~ 1%)
    #     if self.y0_p.bias is not None:
    #         with torch.no_grad():
    #             self.y0_p.bias.fill_(-4.6)
        
    #     # Set Sigma bias = 0.5 (để sigma ban đầu ổn định)
    #     if self.y0_sigma.bias is not None:
    #          with torch.no_grad():
    #             self.y0_sigma.bias.fill_(0.5)

    #     # --- TREATMENT HEADS ---
    #     # Tương tự cho Treatment
    #     if self.y1_p.bias is not None:
    #         with torch.no_grad():
    #             self.y1_p.bias.fill_(-4.6)
                
    #     if self.y1_sigma.bias is not None:
    #          with torch.no_grad():
    #             self.y1_sigma.bias.fill_(0.5)
                
    def forward(self, inputs):
        z = self.shared(inputs)
        t_pred = torch.sigmoid(self.treat_out(z))
        
        # --- CONTROL FLOW ---
        # Đi qua thân chung
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
        
        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps

def dragonnet_loss(y_t, y_c, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, response_lambda=1.0, uplift_lambda = 1.0):
    
    t_pred_clipped = torch.clamp(t_pred, 0.01, 0.99)
    loss_t = torch.sum(F.binary_cross_entropy(t_pred_clipped, t_true, reduction='none'))
    
    loss_1 = zero_inflated_lognormal_loss(y_t, y1_pred)
    loss_0 = zero_inflated_lognormal_loss(y_c, y0_pred)
    loss_y = (loss_0 +  loss_1)
    # print (f"lossy = {loss_y} ") 

    # print (f"losst = {loss_t} | lossy = {loss_y}")
    loss = loss_y + alpha * loss_t
    
    return loss

def tarreg_loss(y_true, t_true, t_pred, y0_pred_c, y1_pred_t, eps, beta=1.0):
    
    # Targeted regularization
    t_pred_clipped = (t_pred + 0.01) / 1.02

    y0_pred = zero_inflated_lognormal_pred(y0_pred_c)
    y1_pred = zero_inflated_lognormal_pred(y1_pred_t)
    
    y_pred = t_true * y1_pred + (1-t_true) * y0_pred
    
    h= (t_true/t_pred_clipped) - ((1-t_true)/ (1-t_pred_clipped))
    
    y_pert = y_pred + eps * h 
    targeted_regularization = torch.sum((y_true-y_pert)**2)
    
    loss = beta * targeted_regularization
    
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