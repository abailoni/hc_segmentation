import numbers

import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d, conv3d

from inferno.io.transform import Transform

class ApplyMaskToBatch(Transform):
    """Applies a mask to target adn prediction where a passed weight mask is set to zero."""
    def __init__(self, targets_are_inverted=True,
                 **super_kwargs):
        super(ApplyMaskToBatch, self).__init__(**super_kwargs)
        self.targets_are_inverted = targets_are_inverted


    def batch_function(self, tensors):
        """
        :param tensors: prediction, target, weight_mask. All with the same shape
        """
        assert len(tensors) == 3
        prediction, target, weight_mask = tensors
        # validate the prediction
        assert prediction.dim() in [4, 5], prediction.dim()
        assert target.size() == prediction.size(), "{}, {}".format(target.size(), prediction.size())
        assert target.size() == weight_mask.size(), "{}, {}".format(target.size(), weight_mask.size())

        # Apply mask:
        zero_mask = (weight_mask == 0).float()
        prediction = prediction * zero_mask

        if self.targets_are_inverted:
            target = 1 - target
            target = target * zero_mask
            target = 1 - target
        else:
            target = target * zero_mask

        return prediction, target