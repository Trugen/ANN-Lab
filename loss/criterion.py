# unchecked
from jittor import nn, Module
from .loss import OhemCrossEntropy2d
from .lovasz_losses import lovasz_softmax

class CriterionDSN(Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = nn.interpolate(X=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = nn.interpolate(X=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = nn.interpolate(X=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss

class CriterionOhemDSN(Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.interpolate(X=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = nn.interpolate(X=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4


class CriterionOhemDSN2(Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.interpolate(X=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        loss2 = lovasz_softmax(nn.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)

        return loss1 + loss2
