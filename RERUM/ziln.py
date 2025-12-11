import torch.nn as nn
import torch
import torch.distributions as tdist
from torch.distributions import LogNormal
import torch.nn.functional as F

class ZILNLoss(nn.Module):
    def __init__(self):
        super(ZILNLoss, self).__init__()
    def forward(self, y, pred):
        """"
        Calculate ZILNLoss
        Arguments:
            pred = output of models, shape (N,3)
                    pred[:, 0]: accuracy(p), pred[:, 1]: mean_log , pred[:, 2]: deviation_log

            y =  real "spend",  shape (N, 1)

        Returns:
            ziln loss value
        """
        # 1. Get params from pred
        positive = (y > 0).float()
        positive_logits = pred[:, 0:1]
        
        # Classification loss using binary cross entropy with logits
        classification_loss = F.binary_cross_entropy_with_logits(
            positive_logits, positive, reduction='mean')
        
        # 2. Regression parameters
        loc = pred[:, 1:2]
        scale = torch.max(
            F.softplus(pred[:, 2:3]),
            torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))
        
        # 3. Safe labels: replace zeros with ones to avoid log(0)
        safe_labels = positive * y + (1 - positive) * torch.ones_like(y)
        
        # 4. Calculate log probability
        log_normal_dist = torch.distributions.LogNormal(loc=loc, scale=scale)
        log_prob = log_normal_dist.log_prob(safe_labels)
        
        # 5. Regression loss (only for positive values)
        regression_loss = -torch.mean(positive * log_prob, dim=-1)
        
        # 6. Overall loss
        overall_loss = classification_loss + regression_loss
        
        return overall_loss
def compute_expected_value(pred):
    """
    Chuyển đổi đầu ra của mô hình (p, mu, sigma) thành giá trị doanh thu dự đoán.
    Công thức: E[y] = p_buy * exp(mu + sigma^2 / 2)
    """
    # 1. Trích xuất tham số (Giống hệt trong hàm Loss)
    p_buy = torch.sigmoid(pred[:, 0])    
    mu = pred[:, 1]                    
    sigma = torch.nn.functional.softplus(pred[:, 2])
    expected_positive_value = torch.exp(mu + 0.5 * sigma**2)

    final_pred = p_buy * expected_positive_value
    
    return final_pred