# coding:utf-8
from __future__ import absolute_import

import os
import os.path

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import RandomErasing

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

'''
Modified from torchvision.datasets.ImageFolder
'''


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, over_sample, class_to_idx, is_train=False):
    """
    Add "label shuffling" function when call "make_dataset()"
    Add "oversample" function to some specific classes
    """

    MAX_LENGTH = 862

    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        SAMPLING = False
        # target is class name. index is the label.
        index = class_to_idx[target]
        if index in over_sample:
            SAMPLING = True

        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        if is_train:
            filenames_in_this_class = os.listdir(d)
            supplement = MAX_LENGTH - len(filenames_in_this_class)
            for i in range(supplement):
                idx = np.random.randint(0, len(filenames_in_this_class))
                fname = filenames_in_this_class[idx]
                path = os.path.join(d, fname)
                item = (path, class_to_idx[target])
                images.append(item)

        if SAMPLING:
            # If over sample, Just do it again.
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


def enhance_transform(img):
    # The param refernced to default setting in : https://github.com/facebook/fc.resnet.torch
    brt = ImageEnhance.Brightness(img)
    random_brt = 1 + (np.random.uniform() - 0.5) * 0.8  # (0.6, 1.4) uniform
    img = brt.enhance(random_brt)

    color = ImageEnhance.Color(img)
    img = color.enhance(1 + (np.random.uniform() - 0.5) * 0.8)  # (0.6, 1.4) uniform

    contrast = ImageEnhance.Contrast(img)
    random_cts = 1 + (np.random.uniform() - 0.5) * 0.8  # x belongs to(0.6, 1.4), p(x<1) = p(x>1)
    img = contrast.enhance(random_cts)

    '''
    # 锐化
    sharpness = ImageEnhance.Sharpness(img)
    random_sharp = (np.random.uniform() - 0.5) * 0.5  # (-0.25, 0.25) uniform
    img = sharpness.enhance(random_sharp)
    '''

    '''TODO: randomly rotate.But I think it doesn't matter.'''
    rotate_ang = (np.random.uniform() - 0.5) * 20  # (-10.0, 10.0) uniform
    img = img.rotate(rotate_ang)
    return img


class SceneDataset(Dataset):
    def __init__(self, over_sample=[], batch_size=64, val_batch_size=16, img_size=224, is_train=True,
                 image_enhance=False):
        self.train_path = './data/train'
        self.val_path = './data/val'
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.img_size = img_size
        self.is_train = is_train
        self.over_sample = over_sample
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandomErasing()  # https://github.com/zhunzhong07/Random-Erasing
            ]),
            'val': transforms.Compose([
                transforms.Scale(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if image_enhance:
            self.transforms['train'] = transforms.Compose([
                transforms.RandomSizedCrop(self.img_size),
                transforms.Lambda(enhance_transform),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                RandomErasing()
            ])
        print("Build a Dataset: Image Enhance:{}".format(image_enhance))

    def get_dataloader(self):
        transformed_dataset_train = OverSampleDataset(root=self.train_path,
                                                      over_sample=self.over_sample,
                                                      transform=self.transforms['train'],
                                                      is_train=True)
        transformed_dataset_val = OverSampleDataset(root=self.val_path,
                                                    transform=self.transforms['val'],
                                                    is_train=False)

        dataloader = {
            'train': DataLoader(transformed_dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=5),
            'val': DataLoader(transformed_dataset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=5)
        }
        if self.is_train:
            return dataloader['train']
        else:
            return dataloader['val']

    def train(self, is_train=True):
        self.is_train = is_train
        return self

    def eval(self):
        self.is_train = False
        return self

    def __len__(self):
        dataset_sizes = {'train': 862 * 80, 'val': 7120}
        if self.is_train:
            return dataset_sizes['train']
        else:
            return dataset_sizes['val']


class OverSampleDataset(Dataset):
    def __init__(self, root, over_sample=[], transform=None, target_transform=None,
                 loader=default_loader, is_train=False):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, over_sample, class_to_idx, is_train)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.is_train = is_train
        self.root = root
        self.over_sample = over_sample
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if len(over_sample) != 0:
            print('Over sample index:{}'.format(self.over_sample))

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)