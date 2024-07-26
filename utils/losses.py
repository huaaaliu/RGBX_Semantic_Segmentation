import torch
import torch.nn as nn

from . import base
from . import functional as F
from . import _modules as modules


class JaccardLoss(base.Loss):

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = modules.Activation(activation, dim=1)
        self.per_image = per_image
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False,
                 # smoothing=0.,
                 aux_loss_weight=0, aux_loss_thres=50, **kwargs):  # TODO add for more loss
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty
        #
        # self.smoothing = smoothing
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_thres = aux_loss_thres

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        #
        if self.aux_loss_weight > 0:
            gt = torch.sum(y_gt, axis=[2, 3], keepdim=True)
            gt = (gt > self.aux_loss_thres).type(gt.dtype)

            if y_pr.shape[1] > 1:
                pr = torch.argmax(y_pr, axis=1, keepdim=True)
            else:
                pr = (y_pr > 0.5)
            pr = pr.type(y_pr.dtype)
            pr = torch.sum(pr, axis=[2, 3], keepdim=True)
            pr = (pr > self.aux_loss_thres).type(pr.dtype)

            class_loss = F.binary_crossentropy(
                pr, gt,
                # pos_weight=self.pos_weight,
                # neg_weight=self.neg_weight,
                # label_smoothing=self.label_smoothing,
            )
            class_loss = class_loss.mean()

        # if self.smoothing > 0:  # dice loss not smooth
        #     y_gt = y_gt * (1-self.smoothing)
        #     y_gt = y_gt + self.smoothing / y_gt.shape[1]

        dice_loss = 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )

        if self.aux_loss_weight > 0:
            return dice_loss * (1 - self.aux_loss_weight) + class_loss * self.aux_loss_weight

        return dice_loss


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(base.Loss):

    def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None, scale=1):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.scale = scale

    def forward(self, pr, gt):
        if len(gt.shape) < len(pr.shape):
            gt = gt.unsqueeze(axis=-1)
        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.scale


class BinaryClassBCELoss(base.Loss):  #

    def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        # TODO cal whole image class label

        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, class_weights=None, logits=False, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None]
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        if self.logits:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pr, gt, reduction='none')
        else:
            bce_loss = F.binary_crossentropy(pr, gt, label_smoothing=self.label_smoothing)

        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()

        return focal_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalDiceLoss(base.Loss):

    def __init__(self, lamdba=2):
        super().__init__()
        self.lamdba = lamdba
        self.focal = BinaryFocalLoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return self.lamdba * self.focal(y_pred, y_true) + self.dice(y_pred, y_true)


class BCEDiceLoss(base.Loss):

    def __init__(self, lamdba=2):
        super().__init__()
        self.lamdba = lamdba
        self.bce = BCELoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        # print(y_pred[0,1,:,:])
        # print(torch.max(y_pred[0,1,:,:]))
        # print(torch.min(y_pred[0, 1, :, :]))
        # print(torch.mean(y_pred[0, 1, :, :]))
        # print(y_true[3,:,:])
        # print(torch.max(y_true[3, :, :]))
        # print(torch.min(y_true[3, :, :]))
        if y_pred.shape[1] > 1:
            y_pred = torch.sigmoid(y_pred)
            # y_pred = torch.sigmoid(torch.softmax(y_pred, dim=1))
            y_pred = torch.unsqueeze(y_pred[:, 1, :, :], dim=1)
        y_true = torch.unsqueeze(y_true, dim=1).float()  # TODO
        y_true[y_true == 255] = 0
        # print(y_pred.shape)
        # print(y_pred.dtype)
        # print(y_true.shape)
        # print(y_true.dtype)
        return self.lamdba * self.bce(y_pred, y_true) + self.dice(y_pred, y_true)
