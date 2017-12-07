# coding=utf-8
from __future__ import absolute_import
from torchvision.transforms import *
import numpy as np
import torch
import torch.optim as optim
import pickle
import os


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_model(model_weight, use_gpu):
    if use_gpu:
        model = torch.load(model_weight, pickle_module=pickle)
        model = model.cuda()
    else:
        model = torch.load(model_weight, map_location=lambda storage, loc: storage, pickle_module=pickle)
    print('Loading checkpoint from %s' % model_weight)
    return model


def get_optimizer(param, learning_rate, optim_name="SGD", weight_decay=1e-4, nesterov=True):
    if optim_name == "Adam":
        optimizer = optim.Adam(param, lr=learning_rate, weight_decay=weight_decay)
    elif optim_name == 'RMSprop':
        optimizer = optim.RMSprop(param, lr=learning_rate, weight_decay=weight_decay, alpha=0.9, eps=1.0, momentum=0.9)
    else:
        optimizer = optim.SGD(param, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


class RandomErasing(object):
    """
    @auth:
        zhunzhong07
    @GitHub:
        https://github.com/zhunzhong07/Random-Erasing
    """
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    # img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    # img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    # img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    # img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img