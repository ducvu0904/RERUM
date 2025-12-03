import torch.nn as nn
import torch
import torch.distributions as tdist
from torch.distributions import LogNormal

class ZILNLoss(nn.Module):
    def __innit__ (self):
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
        #1. Get params from pred
        p = torch.sigmoid(pred[:, 0]).unsqueeze(1)

        mean_log = pred[:, 1].unsqueeze(1)
        devi_log = torch.nn.functional.softplus(pred[:, 2]).unsqueeze(1) +  (1e-6)

        #masking
        y_is_zero = (y < 1e-6)
        y_is_positive = (y>= 1e-6)

        #Calculate classification loss
        when_zero = y_is_zero * torch.log(1-p +1e-6)
        when_positive = y_is_positive * torch.log(p +1e-6)

        classification_loss = - (when_zero + when_positive)

        #Calculate regression loss
        log_normal_dist = LogNormal(mean_log, devi_log)
        log_likelihood = log_normal_dist.log_prob(y+ 1e-6)

        regression_loss = - (y_is_positive * log_likelihood)

        #Overall loss
        overall_loss = classification_loss.mean() + regression_loss.mean()
        return overall_loss
def compute_expected_value(pred):
    """
    Chuyển đổi đầu ra của mô hình (p, mu, sigma) thành giá trị doanh thu dự đoán.
    Công thức: E[y] = p_buy * exp(mu + sigma^2 / 2)
    """
    # 1. Trích xuất tham số (Giống hệt trong hàm Loss)
    p_buy = torch.sigmoid(pred[:, 0])    
    mu = pred[:, 1]                    
    sigma = torch.nn.functional.softplus(pred[:, 2]) + 1e-6

    expected_positive_value = torch.exp(mu + 0.5 * sigma**2)

    final_pred = p_buy * expected_positive_value
    
    return final_pred