import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd.function import Function

class IsLandLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, tlabel, tlambda, size_average=True):
        super(IsLandLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.islandlossFunc = IsLandlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.tlabel = tlabel
        self.tlambda = tlambda

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)

        loss = self.islandlossFunc(feat, self.tlabel, label, self.centers, self.tlambda, batch_size_tensor)
        return loss


class IsLandlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, tlabel, label, centers, tlambda, batch_size):
        ctx.save_for_backward(feature, tlabel, label, centers, tlambda, batch_size)
        centers_batch = centers.index_select(0, label.long())
        islandlosspart = 0.0
        for item1 in label:
            seg_label = tlabel[tlabel!=item1.item()]
            centers_left = centers.index_select(0, seg_label.long())
            element1 = centers.data[int(item1.item())]
            for item2 in centers_left:
                dot_result = element1.dot(item2)
                mod_item1 = element1.norm()
                mod_item2 = item2.norm()
                divide_factor = mod_item1*mod_item2
                islandlosspart += dot_result/divide_factor+1
        islandloss_ts = tlambda*islandlosspart
        center_loss = (feature - centers_batch).pow(2).sum() / 2.0 / batch_size
        return (islandloss_ts+center_loss)

    @staticmethod
    def backward(ctx, grad_output):
        feature, tlabel, label, centers, tlambda, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, None, grad_centers / batch_size, tlambda, None


def main(test_cuda=False):
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")

    y = torch.Tensor([0, 0, 2, 1]).to(device)
    y_total_label = torch.Tensor(range(10)).to(device)
    tlambda = torch.FloatTensor([0.25])
    ct = IsLandLoss(10, 2, y_total_label, tlambda, size_average=True).to(device)

    feat = torch.zeros(4, 2).to(device).requires_grad_()
    print(list(ct.parameters()))
    print(ct.centers.grad)
    out = ct(y, feat)
    print(out.item())
    out.backward()
    print(ct.centers.grad)
    print(feat.grad)

if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=False)