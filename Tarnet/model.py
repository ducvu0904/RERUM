import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, cate_dims, num_count, shared_hidden=200, outcome_hidden=100, shared_dropout = 0.0,  outcome_dropout=0.0):
        super(TarnetBase, self).__init__()
        
        # Create list of embedding layers for categorical features
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, 10) for dim in cate_dims
        ])
        self.num_count = num_count
        
        # Categorical features use embedding; numerical features are concatenated directly.
        total_emb_dim = (len(cate_dims) * 10) + num_count
        
        
        self.shared = nn.Sequential(
        nn.Linear(in_features=total_emb_dim, out_features=shared_hidden),
        nn.ELU(),
        nn.Dropout(shared_dropout),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ELU(),
        nn.Dropout(shared_dropout),
        nn.Linear(in_features=shared_hidden, out_features=shared_hidden),
        nn.ELU(),
        nn.Dropout(shared_dropout)
        )
        
        self.y0 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ELU(),
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ELU(),
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=1)
        )
        
        self.y1 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ELU(),
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ELU(),
        nn.Dropout(outcome_dropout),
        nn.Linear(in_features=outcome_hidden, out_features=1)
        )
               
    def forward(self, x_cat, x_num):
        """
        forward method to train model.

        Parameters
        ----------
        x_cat: torch.Tensor
            categorical covariates
        x_num: torch.Tensor
            numerical covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        """
        # Process categorical embeddings and append numerical features directly.
        embeddings = []
        for i, emb_layer in enumerate(self.cat_embeds):
            embeddings.append(emb_layer(x_cat[:, i].long()))
        if self.num_count > 0:
            embeddings.append(x_num.float())

        # [batch, n_cat * 10 + n_num]
        z_input = torch.cat(embeddings, dim=1)
        z = self.shared(z_input)
        y0 = self.y0(z)
        y1 = self.y1(z)
        

        return y0, y1
    
def outcome_loss(y_t, y_c, y0_pred, y1_pred):
        
    loss_1 = torch.mean(torch.square((y_t - y1_pred)))
    loss_0 = torch.mean(torch.square((y_c - y0_pred)))
    loss = (loss_0 +  loss_1)
    
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