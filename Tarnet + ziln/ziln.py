import torch.nn as nn
import torch
import torch.distributions as tdist
from torch.distributions import LogNormal
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc


def zero_inflated_lognormal_pred(logits):
    """Calculates predicted mean of zero inflated lognormal logits.
    Numerically stable version.
    """
    positive_probs = torch.sigmoid(logits[..., :1])
    loc = logits[..., 1:2]
    scale = F.softplus(logits[..., 2:])

    # --- STABILITY FIX ---
    # Clamp the exponent to prevent overflow (exp(88) is ~max float32)
    exponent = loc + 0.5 * scale**2
    exponent = torch.clamp(exponent, max=80.0)

    preds = positive_probs * torch.exp(exponent)
    return preds

def zero_inflated_lognormal_loss(labels, logits):
    """Computes the zero inflated lognormal loss."""
    positive = (labels > 0).float()

    positive_logits = logits[..., :1]
    classification_loss = F.binary_cross_entropy_with_logits(
        positive_logits, positive, reduction='mean')

    loc = logits[..., 1:2]
    scale = torch.max(
        F.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(1e-6))
    )

    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)

    # Use sum/count instead of mean to handle empty batches gracefully
    regression_loss = -torch.sum(positive * log_prob) / (torch.sum(positive) + 1e-6)

    return classification_loss + regression_loss