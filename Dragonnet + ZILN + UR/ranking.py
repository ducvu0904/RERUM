import torch 
from ziln import zero_inflated_lognormal_pred
import torch.nn.functional as F
def uplift_ranking_loss(y_true, t_true, y0_pred, y1_pred, T=5):
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
    
    y_t = y_true[t_true==1].unsqueeze(1)
    y_c = y_true[t_true==0].unsqueeze(1)
    uplift_pred = uplift_pred.reshape(-1)
    uplift_pred = torch.clamp(uplift_pred, -50, 50)
    
    N1 = y_t.shape[0]
    N0 = y_c.shape[0]
    
    # Compute softmax separately for each group
    softmax_uplift = F.softmax(uplift_pred, dim=0)
    softmax_uplift_t = softmax_uplift[t_true==1].unsqueeze(1)
    softmax_uplift_c = softmax_uplift[t_true==0].unsqueeze(1)
    
    uplift_loss = - (N0 + N1) * ((1/N1) * torch.sum(y_t * torch.log(softmax_uplift_t)) - (1/N0) * torch.sum(y_c * torch.log(softmax_uplift_c)))
    loss = uplift_loss
    

    return loss

def response_ranking_loss_log(
    y_t, y1_pred_t, 
    y_c, y0_pred_c,
    max_samples = 200,
    detach_pred = True
):
    device = y1_pred_t.device
    
    if detach_pred:
        y1_pred_t = y1_pred_t.detach()
        y0_pred_c = y0_pred_c.detach()
        
    if y_t.numel() < 2 and y_c.numel() < 2:
        return torch.tensor(0.0, device=device)
    
    def subsample(y, y_pred, S):
        n = y.shape[0]
        if n <=S:
            return y, y_pred
        idx = torch.randperm(n, device=y.device)[:S]
        return y[idx], y_pred[idx]
    
    # get subsample
    y_t_s, y1_pred_t_s = subsample(y_t.view(-1), y1_pred_t.view(-1), max_samples)
    y_c_s, y0_pred_c_s = subsample(y_c.view(-1), y0_pred_c.view(-1), max_samples)
    
    def pairwise_logistic (y, y_pred):
        y = y.view(-1)
        y_pred = y_pred.view(-1)
        
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        
        mask  = diff_y > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y.device)
        return torch.log1p(torch.exp(-diff_pred[mask])).mean()
    # within-group
    loss_within_t = pairwise_logistic(y_t_s, y1_pred_t_s) if y_t_s.numel() > 1 else torch.tensor(0.0, device=device)
    loss_within_c = pairwise_logistic(y_c_s, y0_pred_c_s) if y_c_s.numel() > 1 else torch.tensor(0.0, device=device)

    # cross-group: treated > control
    diff_y = y_t_s.unsqueeze(1) - y_c_s.unsqueeze(0)
    diff_pred = y1_pred_t_s.unsqueeze(1) - y0_pred_c_s.unsqueeze(0)
    mask = diff_y > 0
    if mask.sum() > 0:
        loss_cross = torch.log1p(torch.exp(-diff_pred[mask])).mean()
    else:
        loss_cross = torch.tensor(0.0, device=device)

    return loss_within_t + loss_within_c + loss_cross
    
        