
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



        # return (target*(-1*((1-pred))**2)*torch.log(pred)-(1-target)**4*((pred)**2)*torch.log(1-pred)).mean() #(1-pred)
        alpha = 0.25
        gamma = 2
        pred = pred.clamp(self.eps, 1. - self.eps)
        logits = torch.log(pred/(1-pred))

        loss = self.focal_loss_with_logits(logits, target, alpha, gamma, pred)
        # loss = -1*(((1-logit)**alpha * torch.log(logit) * y) + ((1-y)**beta * logit**alpha * torch.log(1-logit) * (1-y)))




        return loss.mean()



class centerloss(nn.Module):

    def forward(self,pred, gt):
      ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
      '''
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss