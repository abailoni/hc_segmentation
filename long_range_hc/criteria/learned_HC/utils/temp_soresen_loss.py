import torch.nn as nn
from torch.autograd import Variable

from inferno.utils.torch_utils import flatten_samples


import numpy as np
from torch import from_numpy

__all__ = ['SorensenDiceLoss', 'GeneralizedDiceLoss']


# TODO: all these fct should be moved to inferno once they are well tested


class SorensenDiceLossPixelWeights(nn.Module):
    """
    Equivalent to the one in inferno (only for binary classes), but REQUIRE pixel weights.
    """
    def __init__(self, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLossPixelWeights, self).__init__()
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target, weight):
        if not self.channelwise:
            numerator = (input * target * weight).sum()
            denominator = (input * input * weight).sum() + (target * target * weight).sum()
            loss = -2. * (numerator / denominator.clamp(min=self.eps))
        else:
            # TODO This should be compatible with Pytorch 0.2, but check
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)
            weight = flatten_samples(weight)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target * weight).sum(-1)
            denominator = (input * input * weight).sum(-1) + (target * target * weight).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self.eps))

            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss




class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, weight=None, channelwise=True, eps=1e-6, auto_weight=False):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        auto_weight : bool
            Whether to automatically compute the classes weights based on the batch.
            If True, weight should not be given
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        if auto_weight:
            assert weight is None
        self.auto_weight = auto_weight
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2. * (numerator / denominator.clamp(min=self.eps))
        else:
            # TODO This should be compatible with Pytorch 0.2, but check
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self.eps))
            if self.weight is not None:
                # With pytorch < 0.2, channelwise_loss.size = (C, 1).
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                # Wrap weights in a variable
                weight = Variable(self.weight, requires_grad=False)
                # Apply weight
                channelwise_loss = weight * channelwise_loss
            elif self.auto_weight:
                sum_targets = target.sum(-1)
                class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)
                # print("Loss without weights",channelwise_loss.sum().cpu().data.numpy())
                channelwise_loss = class_weigths * channelwise_loss * 1.0e12
                # print("Loss with weights", channelwise_loss.sum().cpu().data.numpy())
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss




class GeneralizedDiceLoss(nn.Module):
    """
    Computes the scalar Generalized Dice Loss defined in https://arxiv.org/abs/1707.03237

    This version works for multiple classes and expects predictions for every class (e.g. softmax output) and
    one-hot targets for every class.
    """
    def __init__(self, weight=None, channelwise=False, eps=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        """
        input: torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs:
            - if not channelwise: (batch_size, nb_classes, ...)
            - if channelwise:     (batch_size, nb_channels, nb_classes, ...)
        """
        assert input.size() == target.size()
        if not self.channelwise:
            # Flatten input and target to have the shape (nb_classes, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)

            # Find classes weights:
            sum_targets = target.sum(-1)
            class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)

            # Compute generalized Dice loss:
            numer = ((input * target).sum(-1) * class_weigths).sum()
            denom = ((input + target).sum(-1) * class_weigths).sum()

            loss = 1. - 2. * numer / denom.clamp(min=self.eps)
        else:
            def flatten_and_preserve_channels(tensor):
                tensor_dim = tensor.dim()
                assert  tensor_dim >= 3
                num_channels = tensor.size(1)
                num_classes = tensor.size(2)
                # Permute the channel axis to first
                permute_axes = list(range(tensor_dim))
                permute_axes[0], permute_axes[1], permute_axes[2] = permute_axes[1], permute_axes[2], permute_axes[0]
                permuted = tensor.permute(*permute_axes).contiguous()
                flattened = permuted.view(num_channels, num_classes, -1)
                return flattened

            # Flatten input and target to have the shape (nb_channels, nb_classes, N)
            input = flatten_and_preserve_channels(input)
            target = flatten_and_preserve_channels(target)

            # Find classes weights:
            sum_targets = target.sum(-1)
            class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)

            # Compute generalized Dice loss:
            numer = ((input * target).sum(-1) * class_weigths).sum(-1)
            denom = ((input + target).sum(-1) * class_weigths).sum(-1)

            channelwise_loss = 1. - 2. * numer / denom.clamp(min=self.eps)

            if self.weight is not None:
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                channel_weights = Variable(self.weight, requires_grad=False)
                assert channel_weights.size() == channelwise_loss.size(), "`weight` should have shape (nb_channels, ), `target` should have shape (batch_size, nb_channels, nb_classes, ...)"
                # Apply channel weights:
                channelwise_loss = channel_weights * channelwise_loss

            loss = channelwise_loss.sum()

        return loss



# class GeneralizedDiceLoss(nn.Module):
#     """
#     Computes the scalar Generalized Dice Loss defined in https://arxiv.org/abs/1707.03237
#
#     This version works for multiple classes and expects predictions for every class (e.g. softmax output) and
#     0-1 targets for every class.
#     At the moment a target with multiple "true" classes is also possible, e.g. [1, 1, 0, 0, 0]
#     (not sure it makes sense, but useful for different offsets when more types of offsets can be "on" at the same time.)
#     """
#     def __init__(self, channelwise=True, eps=1e-6, size_average=True):
#         """
#         Parameters
#         ----------
#
#         channelwise : bool
#             Whether to apply the loss channelwise and compute stats for all channels (True)
#             or to compute stats separately for every channel (False).
#         """
#         # TODO: add class/pixel weights
#         super(GeneralizedDiceLoss, self).__init__()
#         self.channelwise = channelwise
#         self.eps = eps
#         self.size_average = size_average
#
#     def forward(self, prediction, target):
#         # if self.use_weights:
#         #     assert len(inputs)==3
#         #     weights = inputs[2]
#         # else:
#         #     assert len(inputs) == 2
#         #     weights = Variable(from_numpy(np.ones_like(target.cpu().data.numpy())))
#
#         if not self.channelwise:
#             raise NotImplementedError()
#         else:
#             # Flatten input and target to have the shape (C, N),
#             # where N is the number of samples
#             prediction = flatten_samples(prediction)
#             target = flatten_samples(target)
#
#
#
#             # Find classes weights:
#             sum_targets = target.sum(-1)
#             class_weigths = 1. / (sum_targets*sum_targets).clamp(min=self.eps)
#
#
#             # # Compute generalized Dice loss:
#             numer = ((prediction*target).sum(-1) * class_weigths).sum()
#             denom = ((prediction+target).sum(-1) * class_weigths).sum()
#
#             loss = 1. - 2. * numer / denom.clamp(min=self.eps)
#
#             if self.size_average:
#                 raise NotImplementedError()
#
#         return loss
#
# class GeneralizedDiceLoss_mod(nn.Module):
#     """
#     Computes the scalar Generalized Dice Loss defined in https://arxiv.org/abs/1707.03237
#
#     This version works for multiple classes and expects predictions for every class (e.g. softmax output) and
#     0-1 targets for every class.
#     At the moment a target with multiple "true" classes is also possible, e.g. [1, 1, 0, 0, 0]
#     (not sure it makes sense, but useful for different offsets when more types of offsets can be "on" at the same time.)
#     """
#
#     def __init__(self, channelwise=True, eps=1e-6, size_average=True):
#         """
#         Parameters
#         ----------
#
#         channelwise : bool
#             Whether to apply the loss channelwise and compute stats for all channels (True)
#             or to compute stats separately for every channel (False).
#         """
#         # TODO: add class/pixel weights
#         super(GeneralizedDiceLoss, self).__init__()
#         self.channelwise = channelwise
#         self.eps = eps
#         self.size_average = size_average
#
#     def forward(self, prediction, target):
#         # if self.use_weights:
#         #     assert len(inputs)==3
#         #     weights = inputs[2]
#         # else:
#         #     assert len(inputs) == 2
#         #     weights = Variable(from_numpy(np.ones_like(target.cpu().data.numpy())))
#
#         if not self.channelwise:
#             raise NotImplementedError()
#         else:
#             # Flatten input and target to have the shape (C, N),
#             # where N is the number of samples
#             prediction = flatten_samples(prediction)
#             target = flatten_samples(target)
#
#             # Find classes weights:
#             sum_targets = target.sum(-1)
#             class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)
#
#             # # Compute generalized Dice loss:
#             numer = ((prediction * target).sum(-1) * class_weigths).sum()
#             denom = ((prediction + target).sum(-1) * class_weigths).sum()
#
#             loss = 1. - 2. * numer / denom.clamp(min=self.eps)
#
#             if self.size_average:
#                 raise NotImplementedError()
#
#         return loss


# class DiceLossMultipleClasses(nn.Module):
#     """
#     Computes the loss scalar defined in https://arxiv.org/abs/1707.03237, with the addition of optional pixel-weights
#     (similar to the CrossEntropy implementation).
#     """
#     def __init__(self, nb_classes=None):
#         super(DiceLossMultipleClasses, self).__init__()
#         self.nb_classes = nb_classes
#
#
#     def forward(self, probs, target):
#         """
#         The "classes" expected axis is the number 1.
#         """
#         assert probs.size() == target.size(), "Input sizes must be equal."
#         if self.nb_classes is not None:
#             assert probs.size()[1]==self.nb_classes
#
#         # uniques = np.unique(target.numpy())
#         # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"
#
#         # probs = F.softmax(input)
#
#         def sum_over_extra_axes(var):
#             var = var.sum(dim=5)
#             var = var.sum(dim=4)
#             var = var.sum(dim=3)
#             var = var.sum(dim=2)
#             return var
#
#         flat_probs = flatten_samples(probs)
#         flat_targs = flatten_samples(target)
#
#
#         num = (flat_probs * flat_targs).sum(-1)
#
#         den1 = probs * probs  # --p^2
#         den1 = sum_over_extra_axes(den1)
#
#         den2 = target * target  # --g^2
#         den2 = sum_over_extra_axes(den2)
#
#         dice = 2. * (num / (den1 + den2))
#         dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
#
#         dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
#
#         return dice_total



class ComputeChannelWeightsBatch(nn.Module):
    """
    Example with two channels:

    weight1 = nb_targets_1 / nb_all_targets
    weight2 = nb_targets_2 / nb_all_targets
    """
    def __init__(self):
        super(ComputeChannelWeightsBatch, self).__init__()

    def forward(self, targets):
        """
        targets: torch.FloatTensor or torch.cuda.FloatTensor
                 Tensor with binary values 0,1 and shape (batch_size, nb_channels, ...)
        """
        # Flatten to (nb_channels, -1):
        flatten_targets = flatten_samples(targets)
        # Compute channel weights:
        target_sum = flatten_targets.sum(-1)
        channel_weights = target_sum / target_sum.sum()
        return channel_weights