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
    def __init__(self, cate_dims, num_count, shared_hidden=200, outcome_hidden=100, shared_dropout = 0.0,  outcome_dropout=0.0):
        super(DragonNetBase, self).__init__()
        
        #Creat list of embedding layers for categorical features
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, 10) for dim in cate_dims
        ])
        
        # Calculate the total input dimension for the shared layers
        total_emb_dim  = (len(cate_dims) * 10) + num_count
        
        self.shared = nn.Sequential(
        nn.Linear(in_features=total_emb_dim, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_dropout),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_dropout),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ReLU(),
        nn.Dropout(shared_dropout)
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
        
    def forward(self, x_cat, x_num):
        """
        forward method to train model.

        Parameters
        ----------
        x_cat: torch.Tensor
            categorical features
        x_num: torch.Tensor
            numerical features

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        """

        #Process categorical embeddings and raw numerical features.
        embeddings = []
        for i, emb_layer in enumerate(self.cat_embeds):
            embeddings.append(emb_layer(x_cat[:, i].long()))
        for i in range(x_num.shape[1]):
            embeddings.append(x_num[:, i].unsqueeze(1))
            
        # Pass concatenated features through shared representation layers.
        z_input = torch.cat(embeddings, dim=1)
        z = self.shared(z_input)
        t_pred = torch.sigmoid(self.treat_out(z))
        y0 = self.y0(z)
        y1 = self.y1(z)
        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps
    


def dragonnet_loss(y_t, y_c, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0):
    
    t_pred_clipped = torch.clamp(t_pred, 0.01, 0.99)
    loss_t =  F.binary_cross_entropy(t_pred_clipped, t_true, reduction='mean')
    
    loss_1 = torch.mean(torch.square((y_t - y1_pred)))
    loss_0 = torch.mean(torch.square((y_c - y0_pred)))
    loss_y = (loss_0 + loss_1)

    loss = loss_y + alpha * loss_t
    
    return loss

def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, beta=1.0):
    
    # Targeted regularization
    t_pred_clipped = (t_pred + 0.01) / 1.02
    
    y_pred = t_true * y1_pred + (1-t_true) * y0_pred
    
    h= (t_true/t_pred_clipped) - ((1-t_true)/ (1-t_pred_clipped))
    
    y_pert = y_pred + eps * h 
    targeted_regularization = torch.mean((y_true-y_pert)**2)
    
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