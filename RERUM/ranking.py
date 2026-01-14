import torch 
from ziln import zero_inflated_lognormal_pred
import torch.nn.functional as F

def pairwise_ranking_loss_with_count(pred_row, true_row, pred_col, true_col, max_samples=200):
    """
    Tính pairwise ranking loss với sampling. 
    
    Args:
        pred_row: predictions cho row samples
        true_row: ground truth cho row samples
        pred_col: predictions cho col samples  
        true_col: ground truth cho col samples
        
    Returns:
        (loss, num_pairs)
    """
    pred_row = pred_row.view(-1)
    true_row = true_row.view(-1)
    pred_col = pred_col.view(-1)
    true_col = true_col.view(-1)
    
    N = pred_row.shape[0]
    M = pred_col.shape[0]
    
    # Sample if too large
    if N > max_samples:
        idx_row = torch.randperm(N, device=pred_row.device)[:max_samples]
        pred_row = pred_row[idx_row]
        true_row = true_row[idx_row]
        N = max_samples
    
    if M > max_samples:
        idx_col = torch.randperm(M, device=pred_col.device)[:max_samples]
        pred_col = pred_col[idx_col]
        true_col = true_col[idx_col]
        M = max_samples
    
    # Compute pairwise differences
    pred_diff = pred_row. unsqueeze(1) - pred_col.unsqueeze(0)  # [N, M]
    true_diff = true_row.unsqueeze(1) - true_col.unsqueeze(0)  # [N, M]
    
    # Only penalize when prediction disagrees with true ranking
    product = pred_diff * true_diff
    mask = (product < 0)
    
    if mask.any():
        loss = ((pred_diff - true_diff) ** 2)[mask]. sum()
        num_pairs = mask. sum().item()
    else:
        loss = torch.tensor(0.0, device=pred_row.device)
        num_pairs = 0
            
    return loss, num_pairs


def cross_group_ranking_loss_with_count(pred_i, true_i, pred_j, true_j, max_samples=200):
    """
    Cross-group ranking loss theo công thức paper: 
    
    diff_1 = pred_i - true_j
    diff_2 = true_i - pred_j
    
    if diff_1 * diff_2 >= 0: loss = 0
    else:  loss = (diff_1 - diff_2)²
    
    Args:
        pred_i: predictions của group i (y1_pred hoặc y0_pred)
        true_i: ground truth của group i
        pred_j: predictions của group j (y0_pred hoặc y1_pred)
        true_j: ground truth của group j
    """
    pred_i = pred_i.view(-1)
    true_i = true_i. view(-1)
    pred_j = pred_j.view(-1)
    true_j = true_j.view(-1)
    
    N = pred_i.shape[0]
    M = pred_j.shape[0]
    
    # Sample if too large
    if N > max_samples:
        idx_i = torch.randperm(N, device=pred_i. device)[:max_samples]
        pred_i = pred_i[idx_i]
        true_i = true_i[idx_i]
        N = max_samples
    
    if M > max_samples:
        idx_j = torch.randperm(M, device=pred_j.device)[:max_samples]
        pred_j = pred_j[idx_j]
        true_j = true_j[idx_j]
        M = max_samples
    
    # Cross-group differences (pred vs true from other group)
    diff_1 = pred_i. unsqueeze(1) - true_j.unsqueeze(0)  # [N, M] - ŷᵢ - yⱼ
    diff_2 = true_i. unsqueeze(1) - pred_j.unsqueeze(0)  # [N, M] - yᵢ - ŷⱼ
    
    # Only penalize when disagreement
    product = diff_1 * diff_2
    mask = (product < 0)
    
    if mask.any():
        loss = ((diff_1 - diff_2) ** 2)[mask].sum()
        num_pairs = mask.sum().item()
    else:
        loss = torch.tensor(0.0, device=pred_i.device)
        num_pairs = 0
        
    return loss, num_pairs


def response_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred, max_samples=200):
    """
    Response ranking loss với intra-group và cross-group comparisons theo paper.
    """
    if isinstance(y_true, torch. Tensor):
        y_true = y_true.view(-1)
    else:
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if isinstance(t_true, torch.Tensor):
        t_true = t_true.view(-1)
    else:
        t_true = torch.tensor(t_true, dtype=torch. float32)
        
    y0_pred = zero_inflated_lognormal_pred(y0_pred)
    y1_pred = zero_inflated_lognormal_pred(y1_pred)
    
    y_true = y_true.reshape(-1)
    t_true = t_true.reshape(-1)
    
    N1 = (t_true == 1).sum().item()
    N0 = (t_true == 0).sum().item()
    
    if N1 < 2 and N0 < 2:
        return torch.tensor(0.0, device=y_true. device, requires_grad=True)
    
    # Separate by treatment/control
    y_t = y_true[t_true == 1]  # Ground truth của treatment group
    y_c = y_true[t_true == 0]  # Ground truth của control group
    
    y1_pred_t = y1_pred[t_true == 1]  # Y1 prediction for treatment
    y0_pred_t = y0_pred[t_true == 1]  # Y0 prediction for treatment (counterfactual)
    
    y1_pred_c = y1_pred[t_true == 0]  # Y1 prediction for control (counterfactual)
    y0_pred_c = y0_pred[t_true == 0]  # Y0 prediction for control
    
    
    # ========== INTRA-GROUP LOSS ==========
    treat_loss, treat_pairs = torch.tensor(0.0, device=y_true.device), 0
    if N1 > 1:
        # Treatment group:  compare y1_pred (factual) with y_true
        treat_loss, treat_pairs = pairwise_ranking_loss_with_count(
            y1_pred_t, y_t,
            y1_pred_t, y_t,
            max_samples=max_samples
        )
        
    control_loss, control_pairs = torch.tensor(0.0, device=y_true.device), 0
    if N0 > 1:
        # Control group: compare y0_pred (factual) with y_true
        control_loss, control_pairs = pairwise_ranking_loss_with_count(
            y0_pred_c, y_c,
            y0_pred_c, y_c,
            max_samples=max_samples
        )
    
    
    # ========== CROSS-GROUP LOSS (theo paper) ==========
    cross_loss_tc, tc_pairs = torch.tensor(0.0, device=y_true.device), 0
    cross_loss_ct, ct_pairs = torch.tensor(0.0, device=y_true.device), 0
    
    if N1 > 0 and N0 > 0:
        # Treatment[i] vs Control[j]: 
        # diff_1 = y1_pred_t[i] - y_c[j]  (ŷᵢ¹ - yⱼ⁰)
        # diff_2 = y_t[i] - y0_pred_c[j]  (yᵢ¹ - ŷⱼ⁰)
        cross_loss_tc, tc_pairs = cross_group_ranking_loss_with_count(
            pred_i=y1_pred_t,  # Treatment prediction (factual)
            true_i=y_t,        # Treatment ground truth
            pred_j=y0_pred_c,  # Control prediction (factual)
            true_j=y_c,        # Control ground truth
            max_samples=max_samples
        )
        
        # Control[j] vs Treatment[i] (ngược lại):
        # diff_1 = y0_pred_c[j] - y_t[i]  (ŷⱼ⁰ - yᵢ¹)
        # diff_2 = y_c[j] - y1_pred_t[i]  (yⱼ⁰ - ŷᵢ¹)
        cross_loss_ct, ct_pairs = cross_group_ranking_loss_with_count(
            pred_i=y0_pred_c,  # Control prediction (factual)
            true_i=y_c,        # Control ground truth
            pred_j=y1_pred_t,  # Treatment prediction (factual)
            true_j=y_t,        # Treatment ground truth
            max_samples=max_samples
        )
    
    
    # ========== NORMALIZATION ==========
    total_loss = treat_loss + control_loss + cross_loss_tc + cross_loss_ct
    total_pairs = treat_pairs + control_pairs + tc_pairs + ct_pairs
    
    if total_pairs > 0:
        normalized_loss = total_loss / total_pairs
    else:
        normalized_loss = torch.tensor(0.0, device=y_true.device, requires_grad=True)

    return normalized_loss


def uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
    """Listwise uplift ranking loss"""
    y0_pred = zero_inflated_lognormal_pred(y0_pred)
    y1_pred = zero_inflated_lognormal_pred(y1_pred)
    uplift_pred = y1_pred - y0_pred
    
    if isinstance(y_true, torch. Tensor):
        y_true = y_true.view(-1)
    else:
        y_true = torch.tensor(y_true, dtype=torch.float32)

    if isinstance(t_true, torch.Tensor):
        t_true = t_true.view(-1)
    else:
        t_true = torch.tensor(t_true, dtype=torch. float32)
    
    t_true = t_true.reshape(-1)
    y_true = y_true.reshape(-1) 
    uplift_pred = uplift_pred.reshape(-1)
      
    # Compute log_softmax separately for each group
    log_softmax_uplift_pred = F.log_softmax(uplift_pred, dim=0)
    softmax_uplift_pred_t = log_softmax_uplift_pred[t_true == 1]
    softmax_uplift_pred_c = log_softmax_uplift_pred[t_true == 0]
    
    # Ground truth
    y_t = y_true[t_true == 1]
    y_c = y_true[t_true == 0]
    
    N1 = (t_true == 1).sum().item()
    N0 = (t_true == 0).sum().item()
    
    if N1 == 0 or N0 == 0:
        print(f"⚠️ Warning:  Batch has N1={N1}, N0={N0}. Skipping uplift ranking loss.")
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    loss = -((1/N1) * torch.sum(y_t * softmax_uplift_pred_t + 1e-8) - (1/N0) * torch.sum(y_c * softmax_uplift_pred_c))
    return loss