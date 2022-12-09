import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        return num_objects, info

    def load_single_image(self,video,f):
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            # num_objects = torch.LongTensor([int(1)])
            return Fs, Ms#, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            return Fs, Ms


class YouTube_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root):
        self.root = root
        self.mask_dir = os.path.join(root, 'valid/Annotations')
        # self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'valid/JPEGImages')
        # _imset_dir = os.path.join(root, 'ImageSets')
        # _imset_f = os.path.join(_imset_dir, imset)
        self.K = 11
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.frames_list = {}

        vid_list = sorted(os.listdir(self.image_dir))

        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.num_frames[vid] = len(frames)
            self.frames_list[vid] = frames
            self.videos.append(vid)
            mask = os.listdir(os.path.join(self.mask_dir, vid))
            # print('video:{}, num_mask:{}'.format(vid, len(mask)))
            first_mask = mask[0]
            _mask = np.array(Image.open(os.path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)
        # with open(os.path.join(_imset_f), "r") as lines:
        #     for line in lines:
        #         _video = line.rstrip('\n')
        #         self.videos.append(_video)
        #         self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
        #         _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
        #         self.num_objects[_video] = np.max(_mask)
        #         self.shape[_video] = np.shape(_mask)
        #         _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
        #         self.size_480p[_video] = np.shape(_mask480)
    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size'] = self.shape[video]
        vid_gt_path = os.path.join(self.mask_dir, video)
        frames = self.frames_list[video]
        info['frames_list'] = frames
        obj_frame_idx = []
        obj_num = []
        for i, f in enumerate(frames):
            mask_file = os.path.join(vid_gt_path, f.replace('.jpg', '.png'))
            if os.path.exists(mask_file):
                mask= np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                this_labels = np.unique(mask)
                this_labels = this_labels[this_labels != 0]
                obj_num.extend(this_labels)
                obj_frame_idx.append(i)

        info['obj_frame_idx'] = obj_frame_idx # index of the first frame of multi-object e.g.: [0,2,4]
        num_objects = torch.LongTensor([int(max(obj_num))])
        return num_objects, info

    def load_single_image(self, video, f, obj_frame_idx):
        N_frames = np.empty((1,) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((1,) + self.shape[video], dtype=np.uint8)
        frames = self.frames_list[video]
        frame_name = frames[f]

        img_file = os.path.join(self.image_dir, video, frame_name)
        mask_file = os.path.join(self.mask_dir, video, frame_name.replace('.jpg', '.png'))

        N_frames[0] = np.array(Image.open(img_file).convert('RGB')) / 255.

        if os.path.exists(mask_file):
            if f in obj_frame_idx:
                N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        else:
            N_masks[0] = 255

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()

        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        return Fs, Ms


if __name__ == '__main__':
    pass
