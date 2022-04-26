'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing'
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020
'''

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
import imgaug.augmenters as iaa
import skimage.feature

# face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40, 40), per_channel=True),  # Add color
    iaa.GammaContrast(gamma=(0.5, 1.5))  # GammaContrast with a gamma of 0.5 to 1.5
])


def list2colmatrix(pts_list):
    """
        convert list to column matrix
    Parameters:
    ----------
        pts_list:
            input list
    Retures:
    -------
        colMat:

    """
    assert len(pts_list) > 0
    colMat = []
    for i in range(len(pts_list)):
        colMat.append(pts_list[i][0])
        colMat.append(pts_list[i][1])
    colMat = np.matrix(colMat).transpose()
    return colMat


def find_tfrom_between_shapes(from_shape, to_shape):
    """
        find transform between shapes
    Parameters:
    ----------
        from_shape:
        to_shape:
    Retures:
    -------
        tran_m:
        tran_b:
    """
    assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

    sigma_from = 0.0
    sigma_to = 0.0
    cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

    # compute the mean and cov
    from_shape_points = from_shape.reshape(from_shape.shape[0] // 2, 2)
    to_shape_points = to_shape.reshape(to_shape.shape[0] // 2, 2)
    mean_from = from_shape_points.mean(axis=0)
    mean_to = to_shape_points.mean(axis=0)

    for i in range(from_shape_points.shape[0]):
        temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
        sigma_from += temp_dis * temp_dis
        temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
        sigma_to += temp_dis * temp_dis
        cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

    sigma_from = sigma_from / to_shape_points.shape[0]
    sigma_to = sigma_to / to_shape_points.shape[0]
    cov = cov / to_shape_points.shape[0]

    # compute the affine matrix
    s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    u, d, vt = np.linalg.svd(cov)

    if np.linalg.det(cov) < 0:
        if d[1] < d[0]:
            s[1, 1] = -1
        else:
            s[0, 0] = -1
    r = u * s * vt
    c = 1.0
    if sigma_from != 0:
        c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

    tran_b = mean_to.transpose() - c * r * mean_from.transpose()
    tran_m = c * r

    return tran_m, tran_b


def extract_image_chips(img, points, desired_size=256, padding=0, face_scale=1.0):
    """
        crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
        crop_imgs: list, n
            cropped and aligned faces
    """
    crop_imgs = []
    for p in points:
        shape = []
        for k in range(len(p) // 2):
            shape.append(p[2 * k])
            shape.append(p[2 * k + 1])

        if padding > 0:
            padding = padding
        else:
            padding = 0
        # average positions of face points
        mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
        mean_face_shape_y = [0.3119465, 0.3119465, 0.728106, 0.880233, 0.880233]
        # mean_face_shape_x = [0.3405, 0.6751, 0.5009, 0.3718, 0.6452]
        # mean_face_shape_y = [0.3203, 0.3203, 0.5059, 0.6942, 0.6962]
        from_points = []
        to_points = []

        for i in range(len(shape) // 2):
            x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
            y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
            to_points.append([x, y])
            from_points.append([shape[2 * i], shape[2 * i + 1]])
        # convert the points to Mat
        from_mat = list2colmatrix(from_points)
        to_mat = list2colmatrix(to_points)

        # compute the similar transfrom
        tran_m, tran_b = find_tfrom_between_shapes(from_mat, to_mat)

        probe_vec = np.matrix([1.0, 0.0]).transpose()
        probe_vec = tran_m * probe_vec

        scale = np.linalg.norm(probe_vec)
        angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

        from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
        to_center = [0, 0]
        to_center[1] = desired_size * 0.4
        to_center[0] = desired_size * 0.5

        ex = to_center[0] - from_center[0]
        ey = to_center[1] - from_center[1]
        # change scale to scale*1.2 ,so we can increase face_area in picture
        face_scale = np.random.randint(9, 13)
        face_scale = face_scale / 10.0
        rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scale*face_scale )
        rot_mat[0][2] += ex
        rot_mat[1][2] += ey

        chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
        return chips


# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.01, sh=0.05, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']

        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]


        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]  # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)

        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        #new_map_x = map_x / 255.0  # [0,1]
        return {'image_x': new_image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']

        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)

            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            # print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']

        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:, :, ::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)

        map_x = np.array(map_x)

        spoofing_label_np = np.array([0], dtype=np.long)
        spoofing_label_np[0] = spoofing_label

        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(),
                'map_x': torch.from_numpy(map_x.astype(np.float)).float(), 'spoofing_label': spoofing_label}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir, map_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.map_dir = map_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # print(self.landmarks_frame.iloc[idx, 0])
        for temp in range(500):
            videoname = str(self.landmarks_frame.iloc[idx, 1])
            image_path = os.path.join(self.root_dir, videoname + '.avi')
            map_path = os.path.join(self.map_dir, videoname + '.avi')
            frames_total = len([name for name in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, name))])
            if frames_total > 10:
                break
            else:
                idx = idx + 1

        image_x, map_x = self.get_single_image_x(image_path, map_path, videoname)

        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1  # real
            #map_x = np.ones((32, 32))
        else:
            spoofing_label = 0  # fake
            map_x = np.zeros((32, 32))

        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)

        test = sample['map_x']
        return sample


    def get_single_image_x(self, image_path, map_path, videoname):

        frames_total = len([name for name in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, name))])

        # random choose 1 frame
        lenmap = len(os.listdir(map_path))
        mapnameall = os.listdir(map_path)

        for temp in range(1000):
            image_id = np.random.randint(0, lenmap - 1)

            map_name = mapnameall[image_id]
            image_name = map_name[5:]
            points_name = image_name[:-4] + '.jpg.csv'

            # RGB
            points_path = os.path.join(image_path, points_name)
            image_path2 = os.path.join(image_path, image_name)
            image_x = cv2.imread(image_path2)
            if os.path.exists(points_path):
                if image_x is not None:
                    break

        # gray-map
        map_path = os.path.join(map_path, map_name)
        map_x = cv2.imread(map_path, 0)

        points = pd.read_csv(points_path, index_col=0)
        points = points.values.tolist()

        face_scale = torch.randint(9, 13, (1,)).item()
        face_scale = face_scale / 10.0
        image_x = extract_image_chips(image_x, points, 256, 0.37, face_scale)
        map_x = extract_image_chips(map_x, points, 256, 0.37, face_scale)
        # cv2.imwrite("/home/csy/tt/tt.jpg", image_x)
        # cv2.imwrite("/home/csy/tt/tthui.jpg", map_x)

        # image_x = cv2.resize(crop_face_from_scene(image_x_temp, bbox_path, face_scale), (256, 256))
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x)
        #image_x_aug = image_x
        # image_x_aug = cv2.cvtColor(image_x_aug, cv2.COLOR_BGR2HSV)
        ##########lbp#############################################
        # for colour_channel in (0, 1, 2):
        #     image_x_aug[:, :, colour_channel] = skimage.feature.local_binary_pattern(image_x_aug[:, :, colour_channel], 8, 1.0, method='nri_uniform')
        ##########DoG###############################################
        # gau_matrix = np.asarray([[-2 / 28, -5 / 28, -2 / 28], [-5 / 28, 28 / 28, -5 / 28], [-2 / 28, -5 / 28, -2 / 28]])
        # hight, width, channel = image_x_aug.shape
        # for m in range(channel):
        #     for i in range(1, hight - 1):
        #         for j in range(1, width - 1):
        #             image_x_aug[i - 1, j - 1, m] = np.sum(image_x_aug[i - 1:i + 2, j - 1:j + 2, m] * gau_matrix)
        # image_x_aug = image_x_aug.astype(np.uint8)

        map_x = cv2.resize(map_x, (32, 32))
        map_x = np.where(map_x < 1, map_x, 1)

        return image_x_aug, map_x
