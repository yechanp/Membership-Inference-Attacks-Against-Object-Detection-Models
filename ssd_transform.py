import copy
import numpy as np
import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

class _Transform(object):

    def __init__(self, coder, size, mean,option='all'):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data
        if len(bbox):
            
            # 1. Color augmentation
            img = random_distort(img)

            # 2. Random expansion
            if np.random.randint(2):
                img, param = transforms.random_expand(
                    img, fill=self.mean, return_param=True)
                bbox = transforms.translate_bbox(
                    bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

            # 3. Random cropping
            img, param = random_crop_with_bbox_constraints(
                img, bbox, return_param=True)
            bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            label = label[param['index']]

            # 4. Resizing with random interpolatation
            _, H, W = img.shape
            img = resize_with_random_interpolation(img, (self.size, self.size))
            bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

            # 5. Random horizontal flipping
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label
    
class Transform_vanilla(object):

    def __init__(self, coder, size, mean,option='all'):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data
        
       


        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))


        
        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


# def label_noise(boxes,sigma=0.1):
#         noisy_boxes = []
#         for b in range(len(boxes)):
#             scale_len = np.array([ [item[2]-item[0],item[3]-item[1],item[2]-item[0],item[3]-item[1]] for item in boxes[b] ])
#             eps = np.random.random((boxes[b].shape)) * sigma * scale_len 
#             noisy_train_box =  boxes[b] + eps 
#             noisy_boxes.append(noisy_train_box)
#         return np.array(noisy_boxes)
def label_noise(box,sigma=0.1):
        scale_len = np.array([ [item[2]-item[0],item[3]-item[1],item[2]-item[0],item[3]-item[1]] for item in box ])
        eps = np.random.random((box.shape)) * sigma * scale_len 
        noisy_train_box =  box + eps 
        return np.array(noisy_train_box)
class _Transform_noisy(object):

    def __init__(self, coder, size, mean,sigma=0.1,option='all'):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean
        self.sigma = sigma

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data
        if len(bbox):
            
            # 0. Random Noise on bbox:
            bbox = label_noise(bbox,sigma=self.sigma)
            
            # 1. Color augmentation
            img = random_distort(img)

            # 2. Random expansion
            if np.random.randint(2):
                img, param = transforms.random_expand(
                    img, fill=self.mean, return_param=True)
                bbox = transforms.translate_bbox(
                    bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

            # 3. Random cropping
            img, param = random_crop_with_bbox_constraints(
                img, bbox, return_param=True)
            bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            label = label[param['index']]

            # 4. Resizing with random interpolatation
            _, H, W = img.shape
            img = resize_with_random_interpolation(img, (self.size, self.size))
            bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

            # 5. Random horizontal flipping
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label