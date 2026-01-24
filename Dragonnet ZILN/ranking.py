import torch 
import torch.nn as nn
from ziln import zero_inflated_lognormal_pred
import torch.nn.functional as F

class response_ranking_loss(nn.Module):
    def __init__(self, max_sample = 200):
        super(response_ranking_loss, self).__init__()
        self.S = max_sample
        
    def get_samples(self, pred, true):
        batch_size = pred.shape[0]
        if batch_size <=self.S:
            return pred, true
        indices = torch.randperm(batch_size, device=pred.device)[:self.S]
        return pred[indices], true[indices]
    
    def compute_within_group_loss(self, pred, true):
        diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
        diff_true = true.unsqueeze(1) - true.unsqueeze(0)
        masking = (diff_pred * diff_true) < 0 
        squared_error = (diff_pred - diff_true) **2
        
        return (squared_error * masking.float()).mean()
    
    def compute_cross_group_loss(self, pred_i, true_i, pred_j, true_j):
        term_1 = pred_i.unsqueeze(1) - true_j.unsqueeze(0)
        term_2 = true_i.unsqueeze(1) - pred_j.unsqueeze(1)
        masking = (term_1 * term_2) < 0  
        squared_error = (term_1 - term_2) **2
        
        return (squared_error * masking.float()).mean()
    
    def forward(self, pred_t, true_t, pred_c, true_c):
    #Sampling
        sample_t_pred, sample_t_true = self.get_samples(pred_t, true_t)
        sample_c_pred, sample_c_true = self.get_samples(pred_c, true_c)
        
    #within group loss 
        #treatment
        loss_tt = self.compute_within_group_loss(sample_t_pred, sample_t_true)  
        #control 
        loss_cc = self.compute_within_group_loss(sample_c_pred, sample_c_true)
        #total 
        loss_wr = loss_tt + loss_cc
        
    #cross group
        # loss_tc = self.compute_cross_group_loss(sample_t_pred, sample_t_true, sample_c_pred, sample_c_true)
        # loss_ct = self.compute_cross_group_loss(sample_c_pred, sample_c_true, sample_t_pred, sample_t_true)
        
        # loss_cr = loss_tc + loss_ct
        
        return loss_wr 
    
def uplift_ranking_loss(uplift_pred_t, y_true_t, uplift_pred_c, y_true_c, temperature=100):
    """Listwise uplift ranking loss"""
    
    uplift_pred_t = uplift_pred_t.view(-1)
    y_true_t = y_true_t.view(-1)
    uplift_pred_c = uplift_pred_c.view(-1)
    y_true_c = y_true_c.view(-1)
    
    all_uplift = torch.cat ([uplift_pred_t, uplift_pred_c], dim=0)
    
    #scaled
    uplift_scaled = all_uplift / temperature
    
    #calculate softmax
    all_log_probs = F.log_softmax(uplift_scaled, dim=0)
    
    n_t = uplift_pred_t.shape[0]
    log_probs_t = all_log_probs[:n_t]
    log_probs_c = all_log_probs[n_t:]
    
    N1 = y_true_t.shape[0]
    N0 = y_true_c.shape[0]
    # 4. Tính Loss
    if n_t > 0:
        term_t = (1/N1) * torch.sum(y_true_t * log_probs_t)
    else:
        term_t = torch.tensor(0.0, device=uplift_pred_t.device)
        
    if uplift_pred_c.shape[0] > 0:
        term_c = (1/N0) * torch.sum(y_true_c * log_probs_c)
    else:
        term_c = torch.tensor(0.0, device=uplift_pred_c.device)
    
    loss = - (term_t - term_c)
    
    return loss
    
    