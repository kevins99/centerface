 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class FocalLoss(nn.Module):

    def __init__(self, gamma=4, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps


    def focal_loss_with_logits(self,logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        
        return (torch.log1p(torch.exp(-torch.abs(logits))) + F.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def forward(self, pred, target):
        alpha = 0.25
        gamma = 2
        pred = pred.clamp(self.eps, 1. - self.eps)
        logits = torch.log(pred/(1-pred))

        loss = self.focal_loss_with_logits(logits, target, alpha, gamma, pred)
        # loss = -1*(((1-logit)**alpha * torch.log(logit) * y) + ((1-y)**beta * logit**alpha * torch.log(1-logit) * (1-y)))




        return loss.mean()
