from __future__ import division
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
import math
import time
import tqdm
import os
import argparse
import sys
import copy

### My libs
from dataset.dataset import DAVIS_MO_Test
from model.model import STM

import warnings
_palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment


def flip_tensor(tensor, dim=0):
    inv_idx = torch.arange(tensor.size(dim) - 1, -1, -1, device=tensor.device).long()
    tensor = tensor.index_select(dim, inv_idx)
    return tensor

def Run_video(dataset, video, num_frames, num_objects, model, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
        # print(to_memorize)
    else:
        raise NotImplementedError
    F_last, M_last = dataset.load_single_image(video, 0)  # F: 3,1,480,854  M:11,1,480,854
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last
    F_last = F_last.cuda()
    E_last = E_last.cuda()
    pred = np.zeros((num_frames, M_last.shape[3], M_last.shape[4]))

    # flip evaluation
    # F_last_flip, M_last_flip = dataset.load_single_image_flip(video, 0)  # F: 3,1,480,854  M:11,1,480,854
    # F_last_flip = F_last_flip.unsqueeze(0)
    # M_last_flip = M_last_flip.unsqueeze(0)
    # E_last_flip = M_last_flip
    # F_last_flip = F_last_flip.cuda()
    # E_last_flip = E_last_flip.cuda()

    all_Ms = []
    # temporal_key = 0
    # temporal_val = 0
    # m = 0.7
    for t in range(1, num_frames):

        # memorize
        if t - 1 == 0:
            with torch.no_grad():
                prev_key, prev_value = model(F_last[:, :, 0], E_last[:, :, 0], torch.tensor([num_objects]))
                # prev_key_flip, prev_value_flip = model(F_last_flip[:, :, 0], E_last_flip[:, :, 0], torch.tensor([num_objects]))
        else:
            with torch.no_grad():
                prev_key, prev_value = model(r4_last, E_last[:, :, 0], torch.tensor([num_objects]))
        if t - 1 == 0:
            this_keys, this_values = prev_key, prev_value  # only prev memory
            # this_keys_flip, this_values_flip = prev_key_flip, prev_value_flip
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
            # this_keys_flip = torch.cat([keys_flip, prev_key_flip], dim=3)
            # this_values_flip = torch.cat([values_flip, prev_value_flip], dim=3)
        del prev_key, prev_value#, prev_key_flip, prev_value_flip

        F_, M_ = dataset.load_single_image(video, t)
        # F_flip, M_flip = dataset.load_single_image_flip(video, t)


        F_ = F_.unsqueeze(0)
        F_ = F_.cuda()

        # F_flip = F_flip.unsqueeze(0)
        # F_flip = F_flip.cuda()

        M_ = M_.unsqueeze(0)
        all_Ms.append(M_.cpu().numpy())
        del M_
        # segment
        with torch.no_grad():
            logit, r4e = model(F_[:, :, 0], this_keys, this_values, torch.tensor([num_objects]))
            # logit_flip = model(F_flip[:, :, 0], this_keys_flip, this_values_flip, torch.tensor([num_objects]))
            # E_flip_ori = F.softmax(logit_flip, dim=1)
            # logit_flip = flip_tensor(logit_flip, 3)
        E = F.softmax(logit, dim=1)
        # E_flip = F.softmax(logit_flip, dim=1)

        # multi-scale
        # E_flip_np = E_flip[0].cpu().numpy()#.astype(np.uint8)
        # E_flip_np = E_flip_np.transpose(1,2,0)
        # E_flip_np = cv2.resize(E_flip_np, dsize=(hw_ori[1], hw_ori[0]), interpolation=cv2.INTER_CUBIC)
        # E_flip_np = E_flip_np.transpose(2, 0, 1)
        # E_flip = torch.from_numpy(E_flip_np).unsqueeze(0)
        # E_flip = E_flip.cuda()
        # pred_1 = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        # pred_1 = Image.fromarray(pred_1).convert('P')
        # pred_1.putpalette(_palette)
        # pred_1.save('pred_1.png')
        # pred_2 = torch.argmax(E_flip[0], dim=0).cpu().numpy().astype(np.uint8)
        # pred_2 = Image.fromarray(pred_2).convert('P')
        # pred_2.putpalette(_palette)
        # pred_2.save('pred_2.png')
        # E_fuse = torch.cat([E, E_flip], dim=0)
        # E_fuse = torch.mean(E_fuse, dim=0)
        del logit#, logit_flip
        # update
        if t - 1 in to_memorize:
            keys, values = this_keys, this_values
            # keys_flip, values_flip = this_keys_flip, this_values_flip
            del this_keys, this_values#, this_keys_flip, this_values_flip
        # pred[t] = torch.argmax(E_fuse, dim=0).cpu().numpy().astype(np.uint8)
        pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        # pred_f = Image.fromarray(pred[t]).convert('P')
        # pred_f.putpalette(_palette)
        # pred_f.save('pred_f.png')
        E_last = E.unsqueeze(2)
        r4_last = r4e
        # F_last = F_
        # E_last_flip = E_flip_ori.unsqueeze(2)
        # F_last_flip = F_flip
    Ms = np.concatenate(all_Ms, axis=2)
    return pred, Ms


def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res


def evaluate(model, Testloader, metric):
    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for V in tqdm.tqdm(Testloader):
        num_objects, info = V
        seq_name = info['name']
        num_frames = info['num_frames']
        print('[{}]: num_frames: {}'.format(seq_name, num_frames))

        pred, Ms = Run_video(Testloader, seq_name, num_frames, num_objects, model, Mem_every=12, Mem_number=None)
        # all_res_masks = Es[0].cpu().numpy()[1:1+num_objects]

        # save masks
        # output_dir = 'davis_res'
        # output_path = os.path.join(output_dir, seq_name)
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        # for i in range(0, num_frames):
        #     # frame_name = frames_list[i]
        #     mask = pred[i].astype(np.uint8)
        #     out_mask = Image.fromarray(mask)
        #     out_mask.putpalette(_palette)
        #     out_mask.save(os.path.join(output_path, str(i).zfill(5) + '.png'))

        all_res_masks = np.zeros((num_objects, pred.shape[0], pred.shape[1], pred.shape[2]))
        for i in range(1, num_objects + 1):
            all_res_masks[i - 1, :, :, :] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:-1, :, :]

        all_gt_masks = Ms[0][1:1 + num_objects]
        all_gt_masks = all_gt_masks[:, :-1, :, :]
        j_metrics_res, f_metrics_res = evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)
            # print("{}, J_M: {}, F_M: {}"%(seq_name, JM,FM ))
            print(seq_name, 'J_M: ', JM, 'F_M: ', FM)

    J, F = metrics_res['J'], metrics_res['F']
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    return g_res


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # Volatile


    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", default='0')
        parser.add_argument("-s", type=str, help="set", default='val')
        parser.add_argument("-y", type=int, help="year", default=17)
        parser.add_argument("-D", type=str, help="path to data", default='data/DAVIS/')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",
                            default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights", default='coco_weights/davis_youtube_resnet50_559999.pth') # davis_youtube_resnet50_739999 coco_pretrained_resnet50_499999
        return parser.parse_args()


    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testloader = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR, SET),
                               single_object=(YEAR == 16))

    # model = nn.DataParallel(STM(args.backbone))
    model = STM(args.backbone, state='test')
    # parameter_num = sum(param.numel() for param in model.parameters())
    # print('model parameter: ', parameter_num)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    print("load weights from: {}".format(pth))
    model.load_state_dict(torch.load(pth))
    metric = ['J', 'F']
    print(evaluate(model, Testloader, metric))
