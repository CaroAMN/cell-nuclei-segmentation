import numpy as np
import torch
import torch.nn as nn
# or just use cross entropy loss from pytorch first 

def make_one_hot(input, num_classes=None):
    """
    code from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/dice_loss.py
    Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Shapes:
        predict: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with predict
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    if num_classes is None:
        num_classes = input.max() + 1
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu().long(), 1)
    return result



class BinaryDiceLoss(nn.Module):

    """
    code from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/dice_loss.py
    
    Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """
    Code from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/dice_loss.py
    
    Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        assert output.shape == target.shape, 'output & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        output = F.softmax(output, dim=1)
        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += (dice_loss)
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss



class FocalLoss(nn.Module):
    """
    Implementation based on: https://github.com/Hsuxu/Loss_ToolBox-PyTorch
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss