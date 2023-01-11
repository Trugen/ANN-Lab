# unchecked
from jittor import nn
from .loss import OhemCrossEntropy2d


class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index) 

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.upsample(img=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)

        return loss

class CriterionOhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, thres=0.6, min_kept=200000):
        super(CriterionOhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        # 1/10 of the pixels within a mini-batch, if we use 2x4 on two cards, it should be 200000
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept)

    def forward(self, preds, target):
        # assert len(preds) == 2
        h, w = target.size(1), target.size(2)
        scale_pred = nn.upsample(img=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        # print('OhemCrossEntropy2d Loss: {}'.format(loss.data.numpy()[0]))
        return loss

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.upsample(img=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = nn.upsample(img=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        # scale_pred = nn.upsample(img=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        # loss3 = self.criterion(scale_pred, target)
        return loss1 + loss2*0.4

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.upsample(img=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = nn.upsample(img=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        # scale_pred = nn.upsample(img=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        # loss3 = self.criterion2(scale_pred, target)
        return loss1 + loss2*0.4