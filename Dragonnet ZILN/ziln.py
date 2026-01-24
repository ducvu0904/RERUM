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
    sigma = torch.nn.functional.softplus(logits[..., 2:]) + 1e-3
    sigma = torch.clamp(sigma, 1e-4, 4.0)
    # E[y|y>0] = exp(mu + sigma^2/2)
    log_mean = torch.clamp(mu + 0.5 * sigma.pow(2), max= 10)
    expected_given_positive = torch.exp(log_mean)

    return p * expected_given_positive


def zero_inflated_lognormal_loss(target, prediction):

        logit_p = prediction[:, 0].unsqueeze(1)
        
        positive_target = (target >0).float()
        
        cls_loss = F.binary_cross_entropy_with_logits(logit_p, positive_target, reduction = "none")
        #masking
        positive_mask = (target > 0).squeeze()
        
        if positive_mask.sum()==0:
            reg_loss_sum = torch.tensor(0.0, device=prediction.device)
        else: 
            target_positive = target[positive_mask]
            mu_positive = prediction[positive_mask, 1].unsqueeze(1)
            raw_sigma_positive = prediction[positive_mask, 2].unsqueeze(1)
            sigma_positive = F.softplus(raw_sigma_positive) + 1e-3
            sigma_positive = torch.clamp(sigma_positive, max=4.0)
            # print (f"sigma positive = {sigma_positive}")
            target_log = torch.log(target_positive)
            
            val_loss = (torch.log(sigma_positive) + 0.5 * torch.pow((target_log - mu_positive)/ sigma_positive, 2))
            reg_loss_sum = val_loss.sum()
            
        # print (f"classification: {cls_loss[:5]} | regression: {reg_loss_sum[:5]}")

        total_loss = (cls_loss.sum() + reg_loss_sum)/prediction.shape[0]
        return total_loss