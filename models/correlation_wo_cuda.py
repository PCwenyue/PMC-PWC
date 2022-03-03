"""
correlation_wo_cuda.py
Computes cross correlation between two feature maps with pytorch (without cuda).
Written by Cheng Feng
Github: https://github.com/Ecalpal
Email: fengcheng00016@163.com
Licensed under the MIT License (see LICENSE for details)
"""

import torch
import torch.nn as nn

def correlation(x1, x2, search_range=4):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in image2.
    Args:
        x1: Level of the feature pyramid of image1
        x2: Warped level of the feature pyramid of image2
        search_range: Search range (maximum displacement)
    """

    pad = nn.ConstantPad2d((search_range,search_range,search_range,search_range),0)
    padded_level = pad(x2).type_as(x2)
    b, c, h, w =x1.size() 

    max_offset = search_range * 2 + 1
    cost_vol = torch.zeros((b,max_offset*max_offset,h,w),dtype=x1.dtype).type_as(x1)

    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = padded_level[:, :, y:y+h, x:x+w]
    
            cost_vol[:,y*max_offset+x,:,:] = torch.mean(x1 * slice,dim=1)
    return cost_vol

    # You can also replace it with the following code, both methods are correct.
    #         cost_vol[:,y*max_offset+x,:,:] = torch.sum(x1 * slice,dim=1)
    # return cost_vol / c
