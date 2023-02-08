import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class LogRatioLoss(nn.Module):
    """Log ratio loss function. """
    def __init__(self):
        super(LogRatioLoss, self).__init__()
        self.pdist = L2dist(2)  # norm 2

    def forward(self, input, latent):
        m = input.size()[0]-1   # #paired
        a_input = input[0]            # anchor
        p_input = input[1:]           # paired

        a_latent = latent[0]
        p_latent = latent[1:]

        # auxiliary variables
        idxs = torch.arange(1, m+1).cuda()
        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)

        epsilon = 1e-6

        dist_input = self.pdist.forward(a_input,p_input)
        dist_latent = self.pdist.forward(a_latent,p_latent)

        log_dist = torch.log(dist_input + epsilon)
        log_gt_dist = torch.log(dist_latent + epsilon)
        diff_log_dist = log_dist.repeat(m,1).t()-log_dist.repeat(m, 1)
        diff_log_gt_dist = log_gt_dist.repeat(m,1).t()-log_gt_dist.repeat(m, 1)

        # uniform weight coefficients 
        wgt = indc.clone().float()
        wgt = wgt.div(wgt.sum())

        log_ratio_loss = (diff_log_dist-diff_log_gt_dist).pow(2)

        loss = log_ratio_loss
        loss = loss.mul(wgt).sum()

        return loss
    
class L2dist(nn.Module):
    def __init__(self, p):
        super(L2dist, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-4 / x1.size(0)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff.reshape(diff.shape[0], -1), self.norm).sum(dim=-1)
        return torch.pow(out + eps, 1. / self.norm)
