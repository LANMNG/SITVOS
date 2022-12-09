from __future__ import division
#torch
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import copy

K=6

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array



def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    #colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)

def To_onehot(mask):
    M = np.zeros((K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for k in range(K):
        M[k] = (mask == k).astype(np.uint8)
    return M

def All_to_onehot(masks):
    Ms = np.zeros((K,  masks.shape[0], masks.shape[1]), dtype=np.uint8)
    Ms = To_onehot(masks)
    return Ms

def rm_outliers(mask_sng):
    point_list = np.argwhere(mask_sng > 0)
    if len(point_list) > 0:
        contours, _ = cv2.findContours(mask_sng, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnt_areas = [cv2.contourArea(cnt) for cnt in contours]

        discard_list = []
        T = 250
        for i, cnt_area in enumerate(cnt_areas):
            if cnt_area < T:
                discard_list.append(i)
        if len(discard_list) > 0:
            for k, point_cor in enumerate(point_list):
                for i in discard_list:
                    flag = cv2.pointPolygonTest(contours[i], (point_cor[1], point_cor[0]), False)
                    if flag >= 0:
                        mask_sng[point_cor[0]][point_cor[1]] = 0
                        # print('clink: {}'.format(k))
        return mask_sng
    elif len(point_list) == 0:
        return mask_sng
def denoise(pred, frames):
    for i in range(1, frames):
        mask = pred[i]
        mask_onehot = All_to_onehot(mask).copy()
        for k in range(1, K):
            mask_sng = mask_onehot[k]
            mask_sng = rm_outliers(mask_sng)
            mask_onehot[k] = mask_sng
        mask = np.argmax(mask_onehot, axis=0)
        pred[i] = mask

    return pred





