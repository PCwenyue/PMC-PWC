## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import numbers
import random

import numpy as np
import torch


def image_random_gamma(image, min_gamma=0.7, max_gamma=1.5, clip_image=False):
    gamma = np.random.uniform(min_gamma, max_gamma)
    adjusted = torch.pow(image, gamma)
    if clip_image:
        adjusted.clamp_(0.0, 1.0)
    return adjusted


class RandomGamma:
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    def __call__(self, image):
        return image_random_gamma(
            image,
            min_gamma=self._min_gamma,
            max_gamma=self._max_gamma,
            clip_image=self._clip_image)


# ------------------------------------------------------------------
# Allow transformation chains of the type:
#   im1, im2, .... = transform(im1, im2, ...)
# ------------------------------------------------------------------
class TransformChainer:
    def __init__(self, list_of_transforms):
        self._list_of_transforms = list_of_transforms

    def __call__(self, *args):
        list_of_args = list(args)
        for transform in self._list_of_transforms:
            list_of_args = [transform(arg) for arg in list_of_args]
        if len(args) == 1:
            return list_of_args[0]
        else:
            return list_of_args


# ------------------------------------------------------------------
# Allow transformation chains of the type:
#   im1, im2, .... = split( transform( concatenate(im1, im2, ...) ))
# ------------------------------------------------------------------
class ConcatTransformSplitChainer:
    def __init__(self, list_of_transforms, from_numpy=True, to_numpy=False):
        self._chainer = TransformChainer(list_of_transforms)
        self._from_numpy = from_numpy
        self._to_numpy = to_numpy

    def __call__(self, *args):
        num_splits = len(args)

        if self._from_numpy:
            concatenated = np.concatenate(args, axis=0)
        else:
            concatenated = torch.cat(args, dim=1)

        transformed = self._chainer(concatenated)

        if self._to_numpy:
            split = np.split(transformed, indices_or_sections=num_splits, axis=0)
        else:
            split = torch.chunk(transformed, num_splits, dim=1)

        return split

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs_list):
        h, w, _ = inputs_list[0].shape
        th, tw = self.size
        if w == tw and h == th:
            if len(inputs_list) == 3:
                return inputs_list[0], inputs_list[1], inputs_list[2]

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs_list[0] = inputs_list[0][y1: y1 + th,x1: x1 + tw]
        inputs_list[1] = inputs_list[1][y1: y1 + th,x1: x1 + tw]
        inputs_list[2] = inputs_list[2][y1: y1 + th,x1: x1 + tw]
        if len(inputs_list) == 4:
            inputs_list[3] = inputs_list[3][y1: y1 + th,x1: x1 + tw]
        elif len(inputs_list) == 6:
            inputs_list[4] = inputs_list[4][y1: y1 + th,x1: x1 + tw]
            inputs_list[5] = inputs_list[5][y1: y1 + th,x1: x1 + tw]
        
        if len(inputs_list) == 3:
            return inputs_list[0], inputs_list[1], inputs_list[2]
        elif len(inputs_list) == 4:
            return inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3]
        elif len(inputs_list) == 6:
            return inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3], inputs_list[4], inputs_list[5]
