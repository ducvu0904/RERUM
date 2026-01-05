import torch.nn as nn
import torch
import torch.distributions as tdist
from torch.distributions import LogNormal
import torch.nn.functional as F
import math


def zero_inflated_lognormal_pred(logits):
    """Calculates predicted mean of zero inflated lognormal logits.

    Arguments:
        logits: [batch_size, 3] tensor of logits.

    Returns:
        preds: [batch_size, 1] tensor of predicted mean.
    """
    positive_probs = torch.sigmoid(logits[..., :1])
    loc = logits[..., 1:2]
    scale = torch.nn.functional.softplus(logits[..., 2:])
    scale = torch.clamp(scale, min= 0.5, max=4.0)
    
    expected_given_positive = torch.exp(loc + 0.5 * scale**2)

    return positive_probs * expected_given_positive


def zero_inflated_lognormal_loss(labels, logits):
    """Computes the zero inflated lognormal loss.

    Arguments:
        labels: True targets, tensor of shape [batch_size, 1].
        logits: Logits of output layer, tensor of shape [batch_size, 3].

    Returns:
        Zero inflated lognormal loss value.
    """
    positive = (labels > 0).float()

    positive_logits = logits[..., :1]
    classification_loss = F.binary_cross_entropy_with_logits(
        positive_logits, positive, reduction='none')

    loc = logits[..., 1:2]
    scale = torch.max(
        torch.nn.functional.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))
    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
    regression_loss = -torch.mean(positive * log_prob, dim=-1)

    return torch.mean(classification_loss) + 0.1 * regression_loss


# Keep old class for backward compatibility
class ZILNLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ZILNLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, y, pred):
        return zero_inflated_lognormal_loss(y, pred)


def compute_expected_value(pred):
    """
    Chuyển đổi đầu ra của mô hình (p, mu, sigma) thành giá trị doanh thu dự đoán.
    Công thức: E[y] = p_buy * exp(mu + sigma^2 / 2)
    """
    return zero_inflated_lognormal_pred(pred).squeeze(-1)