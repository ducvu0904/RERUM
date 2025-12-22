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
        y0 = self.headC_out(y0)

        #outcome treatment
        y1 = F.relu(self.headT_1(z))
        y1 = F.relu(self.headT_2(y1))
        y1 = self.headT_out(y1)
        
        eps = self.epsilon(torch.ones_like(t_pred))
        
        return y0 ,y1 ,t_pred, eps

def uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
    #listwise ranking loss
    y0_pred = compute_expected_value(y0_pred)
    y1_pred = compute_expected_value(y1_pred)
    uplift_pred = y1_pred - y0_pred
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.view(-1)
    else:
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if isinstance(t_true, torch.Tensor):
        t_true = t_true.view(-1)
    else:
        t_true = torch.tensor(t_true, dtype=torch.float32)
    
    t_true = t_true.reshape(-1)
    y_true = y_true.reshape(-1) 
    uplift_pred = uplift_pred.reshape(-1)
      
    # Compute softmax separately for each group
    uplift_pred_t = uplift_pred[t_true==1]
    uplift_pred_c = uplift_pred[t_true==0]
    
    softmax_uplift_pred_t = F.softmax(uplift_pred_t, dim=0)
    softmax_uplift_pred_c = F.softmax(uplift_pred_c, dim=0)
    
    #ground truth
    y_t = y_true[t_true==1]
    y_c = y_true[t_true==0]
    
    N1 = (t_true == 1).sum().item()
    N0 = (t_true == 0).sum().item()
    
    if N1 == 0 or N0 == 0:
        print(f"⚠️ Warning: Batch has N1={N1}, N0={N0}. Skipping uplift ranking loss.")
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    loss = -(N0 +N1) * ((1/N1) * torch.sum(y_t * torch.log(softmax_uplift_pred_t + 1e-8)) - (1/N0) * torch.sum(y_c * torch.log(softmax_uplift_pred_c + 1e-8)))
    return loss

def memory_efficient_ranking_loss(pred_row, target_row, pred_col, target_col, max_samples=1000):
    """
    Tính ranking loss với sampling để tiết kiệm bộ nhớ GPU cho batch lớn.
    Samples random pairs instead of computing all N*M pairs.
    """
    pred_row = pred_row.view(-1)
    target_row = target_row.view(-1)
    pred_col = pred_col.view(-1)
    target_col = target_col.view(-1)
    
    N = pred_row.shape[0]
    M = pred_col.shape[0]
    
    # Sample indices if too large
    if N > max_samples:
        idx_row = torch.randperm(N, device=pred_row.device)[:max_samples]
        pred_row = pred_row[idx_row]
        target_row = target_row[idx_row]
        N = max_samples
    
    if M > max_samples:
        idx_col = torch.randperm(M, device=pred_col.device)[:max_samples]
        pred_col = pred_col[idx_col]
        target_col = target_col[idx_col]
        M = max_samples
    
    # Now compute pairwise loss with sampled data
    pred_diff = pred_row.unsqueeze(1) - pred_col.unsqueeze(0)  # [N, M]
    true_diff = target_row.unsqueeze(1) - target_col.unsqueeze(0)  # [N, M]
    
    product = pred_diff * true_diff
    mask = (product < 0)
    
    if mask.any():
        loss = ((pred_diff - true_diff) ** 2)[mask].sum()
    else:
        loss = torch.tensor(0.0, device=pred_row.device)
            
    return loss

def resposne_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.view(-1)
    else:
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if isinstance(t_true, torch.Tensor):
        t_true = t_true.view(-1)
    else:
        t_true = torch.tensor(t_true, dtype=torch.float32)
        
    y0_pred = compute_expected_value(y0_pred)
    y1_pred = compute_expected_value(y1_pred)
    
    y_true = y_true.reshape(-1)
    t_true = t_true.reshape(-1)
    
    N1 = (t_true == 1).sum().item()
    N0 = (t_true == 0).sum().item()
    
    if N1 < 2 and N0 < 2:
        # print(f"⚠️ Warning: Not enough samples for response ranking. N1={N1}, N0={N0}")
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    y_t = y_true[t_true==1].unsqueeze(1)
    y_c = y_true[t_true==0].unsqueeze(1)
    
    y_t_pred = y1_pred[t_true==1].unsqueeze(1)
    y_c_pred = y0_pred[t_true==0].unsqueeze(1)
    
    # ========== INTRA-GROUP LOSS ==========
    treat_loss = torch.tensor(0.0, device=y_true.device)
    if N1 > 1:
        treat_loss = memory_efficient_ranking_loss(
            y_t_pred, y_t, 
            y_t_pred, y_t
        )
        
    control_loss = torch.tensor(0.0, device=y_true.device)
    if N0 > 1:
        control_loss = memory_efficient_ranking_loss(
            y_c_pred, y_c, 
            y_c_pred, y_c
        )
    
    intra_group_loss = control_loss + treat_loss
    
    # ========== CROSS-GROUP LOSS ==========
    cross_loss = torch.tensor(0.0, device=y_true.device)
    if N1 > 0 and N0 > 0:
        # Treatment vs Control (tc)
        # True diff: y_t - y_c_pred 
        loss_tc = memory_efficient_ranking_loss(
            y_t_pred, y_t,
            y_c, y_c_pred
        )
        
        # Control vs Treatment (ct)
        # True diff: y_c - y_t_pred (theo công thức bạn cung cấp)
        loss_ct = memory_efficient_ranking_loss(
            y_c_pred, y_c,
            y_t, y_t_pred
        )
        
        cross_loss = loss_tc + loss_ct
    
    total_loss = intra_group_loss + cross_loss
    return total_loss


def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, ranking_lambda=1.0):
    
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32, device=t_pred.device)
    if not isinstance(t_true, torch.Tensor):
        t_true = torch.tensor(t_true, dtype=torch.float32, device=t_pred.device)
    
    # Ensure all tensors are 1D to avoid broadcasting issues
    y_true = y_true.view(-1)
    t_true = t_true.view(-1)
    t_pred = t_pred.view(-1)
        
    ziln_loss = ZILNLoss()
    
    t_pred_clipped = (t_pred + 0.01) / 1.02
    propensity_loss = torch.sum(F.binary_cross_entropy(t_pred_clipped, t_true))
    
    ziln_t = ziln_loss(y_true.unsqueeze(1), y1_pred).view(-1)
    ziln_c = ziln_loss(y_true.unsqueeze(1), y0_pred).view(-1)
    loss_t = torch.sum(t_true * ziln_t)
    loss_c = torch.sum((1 - t_true) * ziln_c)
    
    loss_uplift_ranking = uplift_ranking_loss(y_true, t_true,t_pred, y0_pred, y1_pred)
    loss_response_ranking = resposne_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred)
    ranking_loss = ranking_lambda *(10 * loss_uplift_ranking +((1e-4) * loss_response_ranking))
    print (f"Ranking loss: {ranking_loss}")
    loss_y = loss_t + loss_c + ranking_loss

    loss = loss_y + alpha* propensity_loss
    return loss
    
def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, ranking_lambda, alpha=1.0, beta=1.0):
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32, device=t_pred.device)
    if not isinstance(t_true, torch.Tensor):
        t_true = torch.tensor(t_true, dtype=torch.float32, device=t_pred.device)
        
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=alpha, ranking_lambda=ranking_lambda)
    t_pred = (t_pred +0.01)/1.02
    
    y0_pred = compute_expected_value(y0_pred)
    y1_pred = compute_expected_value(y1_pred)
    y_pred = t_true * y1_pred + (1-t_true)*y0_pred
    
    h = (t_true/t_pred) - ((1-t_true)/ (1-t_pred))
    
    y_pert = y_pred + eps*h
    tarreg_loss = torch.sum((y_true - y_pert)**2)
    
    loss = vanilla_loss + beta*tarreg_loss
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