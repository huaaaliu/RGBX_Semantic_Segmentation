import torch


def _ignore_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _take_channels(*xs, take_channels=None):
    if take_channels is None:
        return xs
    else:
        xs = [torch.index_select(x, dim=1, index=torch.tensor(take_channels).to(x.device)) for x in xs]
        return xs


def _take_non_empty(pr, gt, drop_empty=True):
    if drop_empty:
        mask = gt[gt.sum(dim=(1, 2, 3)) > 0]
        gt = gt[mask]
        pr = pr[mask]
    return pr, gt


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _sum(x, per_image=False):
    if per_image:
        return torch.sum(x, dim=(2, 3))
    else:
        return torch.sum(x, dim=(0, 2, 3))


def _average(x, weights=None):
    """"""
    if weights is not None:
        x = x * torch.tensor(weights, dtype=x.dtype, requires_grad=False).to(x.device)

    if x.dim() == 2:
        x = x.mean(dim=0)

    return x.mean()


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None,
        class_weights=None, per_image=False, drop_empty=False, take_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _ignore_channels(pr, gt, ignore_channels=ignore_channels)
    pr, gt = _take_channels(pr, gt, take_channels=take_channels)
    #pr, gt = _take_non_empty(pr, gt, drop_empty=drop_empty)

    # if gt.nelement() == 0:
    #     return 1.

    intersection = _sum(gt * pr, per_image)
    union = _sum(gt, per_image) + _sum(pr, per_image) - intersection
    score = (intersection + eps) / (union + eps)

    if drop_empty:
        agg_mask = gt.sum(dim=(2, 3)) if per_image else gt.sum(dim=(0, 2, 3))
        empty_mask = 1. - (agg_mask > 1).float()
        score = score * empty_mask

    return _average(score, class_weights)


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None,
            class_weights=None, per_image=False, drop_empty=False, take_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _ignore_channels(pr, gt, ignore_channels=ignore_channels)
    pr, gt = _take_channels(pr, gt, take_channels=take_channels)

    if drop_empty:
        pr = pr * (gt.sum(dim=(2, 3), keepdims=True) > 0).float()

    tp = _sum(gt * pr, per_image)
    fp = _sum(pr, per_image) - tp
    fn = _sum(gt, per_image) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    # if drop_empty:
    #     agg_mask = gt.sum(dim=(2, 3)) if per_image else gt.sum(dim=(0, 2, 3))
    #     non_empty_mask = (agg_mask > 1).float()
    #     score = score * non_empty_mask

    return _average(score, class_weights)


def accuracy(pr, gt, threshold=0.5, ignore_channels=None, take_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, take_channels=take_channels)
    pr, gt = _ignore_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr).float()
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _ignore_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _ignore_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


# def binary_crossentropy(pr, gt, eps=1e-7, pos_weight=1., neg_weight=1.):
#     pr = torch.clamp(pr, eps, 1. - eps)
#     gt = torch.clamp(gt, eps, 1. - eps)
#     loss = - pos_weight * gt * pr.log() - neg_weight * (1. - gt) * (1. - pr).log()
#     return loss


def binary_crossentropy(pr, gt, eps=1e-7, pos_weight=1., neg_weight=1., label_smoothing=None):
    if label_smoothing is not None:
        label_smoothing = torch.tensor(label_smoothing).to(gt.device)
        gt = gt * (1. - label_smoothing) + (1. - gt) * label_smoothing
    pr = torch.clamp(pr, eps, 1. - eps)
    gt = torch.clamp(gt, eps, 1. - eps)
    loss = - pos_weight * gt * torch.log(pr/gt) - neg_weight * (1. - gt) * torch.log((1. - pr) / (1. - gt))
    return loss
