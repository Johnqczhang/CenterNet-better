#!/usr/bin/python3
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F


def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


@torch.jit.script
def sigmoid_focal_loss_umich(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float=2,
    beta: float=4
):
    p = torch.sigmoid(inputs)
    pos_inds = targets.eq(1).float()
    neg_inds = 1 - pos_inds
    ce_loss = F.binary_cross_entropy_with_logits(inputs, pos_inds, reduction="none")
    p_t = p * pos_inds + (1 - p) * neg_inds
    y_t = pos_inds + (1 - targets) * neg_inds
    loss = ce_loss * ((1 - p_t) ** alpha) * (y_t ** beta)

    loss = loss.sum()
    return loss
