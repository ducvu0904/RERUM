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
        nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=1),
        nn.Softplus()
        )
        
        self.y1 = nn.Sequential(
        nn.Linear(in_features=shared_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(in_features=outcome_hidden, out_features=1),
        nn.Softplus()
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
    
# def uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
#     #listwise ranking loss

#     uplift_pred = y1_pred - y0_pred
    
#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.view(-1)
#     else:
#         y_true = torch.tensor(y_true, dtype=torch.float32)

#     if isinstance(t_true, torch.Tensor):
#         t_true = t_true.view(-1)
#     else:
#         t_true = torch.tensor(t_true, dtype=torch.float32)
    
#     t_true = t_true.reshape(-1)
#     y_true = y_true.reshape(-1) 
#     uplift_pred = uplift_pred.reshape(-1)
      
#     # Compute softmax separately for each group
#     softmax_uplift_pred  = F.softmax(uplift_pred, dim=0)
#     softmax_uplift_pred_t = softmax_uplift_pred[t_true==1]
#     softmax_uplift_pred_c = softmax_uplift_pred[t_true==0]
#     # uplift_pred_t = uplift_pred[t_true==1]
#     # uplift_pred_c = uplift_pred[t_true==0]
    
#     # softmax_uplift_pred_t = F.softmax(uplift_pred_t, dim=0)
#     # softmax_uplift_pred_c = F.softmax(uplift_pred_c, dim=0)
    
#     #ground truth
#     y_t = y_true[t_true==1]
#     y_c = y_true[t_true==0]
    
#     N1 = (t_true == 1).sum().item()
#     N0 = (t_true == 0).sum().item()
    
#     if N1 == 0 or N0 == 0:
#         print(f"⚠️ Warning: Batch has N1={N1}, N0={N0}. Skipping uplift ranking loss.")
#         return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
#     loss = -((1/N1) * torch.sum(y_t * torch.log(softmax_uplift_pred_t + 1e-8)) - (1/N0) * torch.sum(y_c * torch.log(softmax_uplift_pred_c + 1e-8)))
#     return loss

# def memory_efficient_ranking_loss(pred_row, target_row, pred_col, target_col, max_samples=2500):
#     """
#     Tính ranking loss với sampling để tiết kiệm bộ nhớ GPU cho batch lớn.
#     Samples random pairs instead of computing all N*M pairs.
#     """
#     pred_row = pred_row.view(-1)
#     target_row = target_row.view(-1)
#     pred_col = pred_col.view(-1)
#     target_col = target_col.view(-1)
    
#     N = pred_row.shape[0]
#     M = pred_col.shape[0]
    
#     # Sample indices if too large
#     if N > max_samples:
#         idx_row = torch.randperm(N, device=pred_row.device)[:max_samples]
#         pred_row = pred_row[idx_row]
#         target_row = target_row[idx_row]
#         N = max_samples
    
#     if M > max_samples:
#         idx_col = torch.randperm(M, device=pred_col.device)[:max_samples]
#         pred_col = pred_col[idx_col]
#         target_col = target_col[idx_col]
#         M = max_samples
    
#     # Now compute pairwise loss with sampled data
#     pred_diff = pred_row.unsqueeze(1) - pred_col.unsqueeze(0)  # [N, M]
#     true_diff = target_row.unsqueeze(1) - target_col.unsqueeze(0)  # [N, M]
    
#     product = pred_diff * true_diff
#     mask = (product < 0)
    
#     if mask.any():
#         loss = ((pred_diff - true_diff) ** 2)[mask].sum()
#     else:
#         loss = torch.tensor(0.0, device=pred_row.device)
            
#     return loss

# def resposne_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):

#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.view(-1)
#     else:
#         y_true = torch.tensor(y_true, dtype=torch.float32)

#     if isinstance(t_true, torch.Tensor):
#         t_true = t_true.view(-1)
#     else:
#         t_true = torch.tensor(t_true, dtype=torch.float32)
        
    
#     y_true = y_true.reshape(-1)
#     t_true = t_true.reshape(-1)
    
#     N1 = (t_true == 1).sum().item()
#     N0 = (t_true == 0).sum().item()
    
#     if N1 < 2 and N0 < 2:
#         # print(f"⚠️ Warning: Not enough samples for response ranking. N1={N1}, N0={N0}")
#         return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
#     y_t = y_true[t_true==1].unsqueeze(1)
#     y_c = y_true[t_true==0].unsqueeze(1)
    
#     y_t_pred = y1_pred[t_true==1].unsqueeze(1)
#     y_c_pred = y0_pred[t_true==0].unsqueeze(1)
    
#     # ========== INTRA-GROUP LOSS ==========
#     treat_loss = torch.tensor(0.0, device=y_true.device)
#     if N1 > 1:
#         treat_loss = memory_efficient_ranking_loss(
#             y_t_pred, y_t, 
#             y_t_pred, y_t
#         )
        
#     control_loss = torch.tensor(0.0, device=y_true.device)
#     if N0 > 1:
#         control_loss = memory_efficient_ranking_loss(
#             y_c_pred, y_c, 
#             y_c_pred, y_c
#         )
    
#     intra_group_loss = control_loss + treat_loss
    
#     # ========== CROSS-GROUP LOSS ==========
#     cross_loss = torch.tensor(0.0, device=y_true.device)
#     if N1 > 0 and N0 > 0:
#         # Treatment vs Control (tc)
#         # True diff: y_t - y_c_pred 
#         loss_tc = memory_efficient_ranking_loss(
#             y_t_pred, y_t,
#             y_c, y_c_pred
#         )
        
#         # Control vs Treatment (ct)
#         # True diff: y_c - y_t_pred (theo công thức bạn cung cấp)
#         loss_ct = memory_efficient_ranking_loss(
#             y_c_pred, y_c,
#             y_t, y_t_pred
#         )
        
#         cross_loss = loss_tc + loss_ct
    
#     total_loss = intra_group_loss + cross_loss
#     return total_loss

def dragonnet_loss (y_true, t_true, t_pred, y0_pred, y1_pred, alpha=1.0, ranking_lambda =1.0):
    t_pred = (t_pred + 0.01) / 1.02
    propensity_loss = torch.sum(F.binary_cross_entropy(t_pred, t_true))
    
    # print (f"losst = {propensity_loss}")
    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    # loss_uplift_ranking = 10 * uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred)
    # loss_response_ranking = (1e-4) * resposne_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred)
    # print (f"uplift ranking loss = {loss_uplift_ranking}" )
    # print (f" resposne_ranking_loss = {loss_response_ranking}")
    # loss_y = loss0 + loss1 + ranking_lambda * (loss_uplift_ranking + loss_response_ranking)
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
    # scale_factor = torch.mean(y_true**2).detach() + 1e-6
    # normalized_tarreg_loss = targeted_regularization_raw / scale_factor
    # # print (f" tarreg = {normalized_tarreg_loss}")
    return vanilla_loss + beta * targeted_regularization_raw

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