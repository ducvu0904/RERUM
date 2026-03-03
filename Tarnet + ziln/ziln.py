import torch.nn as nn
import torch
import torch.distributions as tdist
from torch.distributions import LogNormal
import torch.nn.functional as F
import numpy as np


def zero_inflated_lognormal_pred(logits):
    """Calculates predicted mean of zero inflated lognormal logits.

    Arguments:
        logits: [batch_size, 3] tensor of logits.

    Returns:
        preds: [batch_size, 1] tensor of predicted mean.
    """

    p = torch.sigmoid(logits[..., :1])
    mu = logits[..., 1:2]
    # Clamp mu to a reasonable range for log(spend): typical range is 0–6 (i.e. spend 1–400)
    # Allowing up to 7 covers outliers; below -1 would mean spend < 0.37 given positive, rare
    mu = torch.clamp(mu, min=-1.0, max=7.0)
    sigma = torch.nn.functional.softplus(logits[..., 2:]) + 1e-3
    # sigma_max = 1.0: limits exp contribution to +0.5 instead of +2.0 at old max
    sigma = torch.clamp(sigma, min=1e-4, max=1.0)
    # E[y] = P(y>0) * E[y|y>0] = sigmoid(p) * exp(mu + sigma^2/2)
    log_mean = torch.clamp(mu + 0.5 * sigma**2, max=8.0)  # tighter clamp
    expected_given_positive = torch.exp(log_mean)

    return p * expected_given_positive  


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
        positive_logits, positive, reduction='mean')

    loc = logits[..., 1:2]
    # Clamp loc (mu) to prevent extreme values that blow up log-prob
    loc = torch.clamp(loc, min=-1.0, max=7.0)
    scale = torch.max(
        F.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps, device=logits.device)))
    # Reduce sigma_max from 2.0 to 1.0 to prevent exp() explosion
    scale = torch.clamp(scale, max=1.0)
    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
    # Normalise over batch size (not just positives) so cls and reg losses stay on the same scale.
    # Dividing by num_positive inflates reg_loss by 1/positive_rate (e.g. ×20 at 5%) and drowns
    # out the classification gradient.
    batch_size = labels.shape[0]
    regression_loss = -(positive * log_prob).sum() / batch_size

    return classification_loss + regression_loss