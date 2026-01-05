import torch 
from ziln import zero_inflated_lognormal_pred
def uplift_ranking_loss(y_true, t_true, t_pred, y0_pred, y1_pred):
    #listwise ranking loss
    y0_pred = zero_inflated_lognormal_pred(y0_pred)
    y1_pred = zero_inflated_lognormal_pred(y1_pred)
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
    softmax_uplift_pred  = F.softmax(uplift_pred, dim=0)
    softmax_uplift_pred_t = softmax_uplift_pred[t_true==1]
    softmax_uplift_pred_c = softmax_uplift_pred[t_true==0]
    # uplift_pred_t = uplift_pred[t_true==1]
    # uplift_pred_c = uplift_pred[t_true==0]
    
    # softmax_uplift_pred_t = F.softmax(uplift_pred_t, dim=0)
    # softmax_uplift_pred_c = F.softmax(uplift_pred_c, dim=0)
    
    #ground truth
    y_t = y_true[t_true==1]
    y_c = y_true[t_true==0]
    
    N1 = (t_true == 1).sum().item()
    N0 = (t_true == 0).sum().item()
    
    if N1 == 0 or N0 == 0:
        print(f"⚠️ Warning: Batch has N1={N1}, N0={N0}. Skipping uplift ranking loss.")
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    
    loss = -((1/N1) * torch.sum(y_t * torch.log(softmax_uplift_pred_t + 1e-8)) - (1/N0) * torch.sum(y_c * torch.log(softmax_uplift_pred_c + 1e-8)))
    return loss

def memory_efficient_ranking_loss(pred_row, target_row, pred_col, target_col, max_samples=200):
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
        
    y0_pred = zero_inflated_lognormal_pred(y0_pred)
    y1_pred = zero_inflated_lognormal_pred(y1_pred)
    
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
            y_c_pred, y_c
        )
        
        # Control vs Treatment (ct)
        # True diff: y_c - y_t_pred (theo công thức bạn cung cấp)
        loss_ct = memory_efficient_ranking_loss(
            y_c_pred, y_c,
            y_t_pred, y_t
        )
        
        cross_loss = loss_tc + loss_ct
    
    total_loss = intra_group_loss + cross_loss
    return total_loss