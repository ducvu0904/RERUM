import torch.nn as nn
import torch
import torch.distributions as tdist
from torch.distributions import LogNormal
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc


def zero_inflated_lognormal_pred(logits):
    """Calculates predicted mean of zero inflated lognormal logits.

    Arguments:
        logits: [batch_size, 3] tensor of logits.

    Returns:
        preds: [batch_size, 1] tensor of predicted mean.
    """

    p = torch.sigmoid(logits[..., :1])
    mu = logits[..., 1:2]
    scale = torch.nn.functional.softplus(logits[..., 2:])
    scale = torch.clamp(scale, min = 1e-4, max = 5.0)
    log_mean = mu + 0.5 * scale**2
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
    pos_weight = torch.tensor([99.0], device=logits.device)
    classification_loss = F.binary_cross_entropy_with_logits(
        positive_logits, positive, reduction='mean', pos_weight = pos_weight)

    loc = logits[..., 1:2]
    scale = torch.max(
        F.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps, device=logits.device)))
    scale = torch.clamp(scale, min = 1e-4, max = 5.0)
    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
    batch_size = labels.shape[0]
    regression_loss = -(positive * log_prob).sum() / batch_size

    mu_mean = loc.mean().item()
    sigma_mean = scale.mean().item()

    return classification_loss + regression_loss, classification_loss.item(), regression_loss.item(), mu_mean, sigma_mean


def compute_classification_metrics(labels, logits):
    """Computes F1 score and PR-AUC for the classification component of ZILN.

    Uses adaptive threshold based on positive rate for F1
    (more appropriate for imbalanced data than fixed 0.5).

    Arguments:
        labels: True targets, tensor or numpy array of shape [N, 1].
        logits: ZILN logits, tensor or numpy array of shape [N, 3].

    Returns:
        dict with 'f1' and 'pr_auc' keys. Both may be None if only one class present.
    """
    # Convert to numpy if tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    
    # Extract classification logits (first column) and compute probabilities
    p_logits = logits[..., 0]
    probs = 1 / (1 + np.exp(-p_logits))  # sigmoid
    
    # Binary labels: 1 if positive (y > 0), 0 otherwise
    labels_binary = (labels.flatten() > 0).astype(int)
    
    # Handle edge case: only one class present
    unique_classes = np.unique(labels_binary)
    if len(unique_classes) < 2:
        return {'f1': None, 'pr_auc': None}
    
    # Compute PR-AUC
    precision, recall, _ = precision_recall_curve(labels_binary, probs)
    pr_auc = auc(recall, precision)
    
    # Use adaptive threshold based on positive rate for F1
    positive_rate = labels_binary.mean()
    threshold = np.percentile(probs, (1 - positive_rate) * 100)
    threshold = max(threshold, 0.01)  # Minimum to avoid all-positive
    
    preds_binary = (probs >= threshold).astype(int)
    f1 = f1_score(labels_binary, preds_binary, zero_division=0)
    
    return {'f1': f1, 'pr_auc': pr_auc}