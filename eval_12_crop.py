# coding=utf-8
from __future__ import print_function

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logging
import argparse
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from utils import *

batch_size = 64
dataset_size = 7120
use_gpu = torch.cuda.is_available()


def get_data(transform):
    transformed_dataset_val = torchvision.datasets.ImageFolder(root='./data/val',
                                                               transform=transform)
    dataloader = DataLoader(transformed_dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader


def predict(model, dataloader, criterion=nn.CrossEntropyLoss()):
    model.eval()

    running_loss = 0.0
    top_1 = AverageMeter()
    top_3 = AverageMeter()

    raw_results = []
    label_list = []

    step = 0
    for data in tqdm(dataloader):
        inputs, labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        running_loss += loss.data[0]

        acc_1, acc_3 = accuracy(outputs.data, labels.data, top_k=(1, 3))
        acc_1, acc_3 = acc_1[0], acc_3[0]
        top_1.update(acc_1, inputs.data.size(0))
        top_3.update(acc_3, inputs.data.size(0))
        step += 1

        raw_result = [
            {"index": batch_size * step + idx,
             "prob": prob.tolist()
             }
            for idx, prob in zip(range(batch_size), outputs.data)
            ]

        raw_results += raw_result
        label_list += labels.data.tolist()

    epoch_loss = running_loss * batch_size / dataset_size

    print("Val loss:{:1.3f},\ttop_1:{:1.3f},\ttop_3:{:1.3f}".format(
        epoch_loss, top_1.avg, top_3.avg))

    return raw_results, label_list


def prob_add(list_1, list_2):
    if len(list_1) == 0:
        return list_2
    if len(list_2) == 0:
        return list_1

    sum_list = []
    for item1, item2 in zip(list_1, list_2):
        temp = []
        try:
            item1 = item1['prob']
        except:
            pass
        try:
            item2 = item2['prob']
        except:
            pass

        for i in range(80):
            temp.append(item1[i] + item2[i])
        sum_list.append(temp)
    return sum_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model_name', default='resnet50', type=str, help='Model name')
    parser.add_argument('--ckpt', default='./resnet50/resnet50_places365_model_wts_1436.pth',
                        type=str, help='Model name')
    parser.add_argument('--multi_scale', default='224,256,320,384', type=str, help='12 Crop with multi scale')
    parser.add_argument('--crop_size', default=224, type=int, help='crop size')


    args = parser.parse_args()
    logging.info(args)
    time_start = time.time()

    model_name = args.model_name
    model_weight = args.ckpt

    model = get_model(model_weight, use_gpu)
    for param in model.parameters():
        param.requires_grad = False

    img_sizes = [int(s) for s in args.multi_scale]
    crop_size = args.crop_size

    ens_probs = []
    for img_size in img_sizes:
        print("IMG_SIZE:", img_size)
        probs = []
        for pos in range(12):
            print("Crop:", pos)

            def twelve_crop(img):
                if pos % 6 == 5:
                    crop = img.resize((crop_size, crop_size))
                    if pos >= 6:
                        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                    return crop

                X, Y = img.size

                if pos % 6 == 0:
                    center = (crop_size / 2, crop_size / 2)
                if pos % 6 == 1:
                    center = (X - crop_size / 2, crop_size / 2)
                if pos % 6 == 2:
                    center = (X // 2, Y // 2)
                if pos % 6 == 3:
                    center = (crop_size / 2, Y - crop_size / 2)
                if pos % 6 == 4:
                    center = (X - crop_size / 2, Y - crop_size / 2)

                x, y = center[0] - crop_size / 2, center[1] - crop_size / 2

                region = (x, y, x + crop_size, y + crop_size)
                crop = img.crop(region)
                if pos >= 6:
                    crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                return crop

            transform = transforms.Compose([
                transforms.Scale(img_size),
                transforms.Lambda(twelve_crop),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            dataloader = get_data(transform)

            prob, labels = predict(model, dataloader)
            probs = prob_add(probs, prob)

        ens_probs.append(probs)
        ens_pred = [np.argsort(item)[-1:-4:-1] for item in probs]
        top_1, top_3 = 0, 0
        for pred, label in zip(ens_pred, labels):
            if label in pred:
                top_3 += 1
            if label == pred[0]:
                top_1 += 1
        top_1 = top_1 * 100.0 / dataset_size
        top_3 = top_3 * 100.0 / dataset_size
        print("Ensembled: top_1:{:.3f}\t top_3:{:0.3f}".format(top_1, top_3))

        cache_path = "./results/cache/{}_prob_{}.pkl".format(model_name, img_size)
        pd.to_pickle(probs, cache_path)
        pd.to_pickle(labels, './results/cache/{}_labels.pkl'.format(model_name))

    final_prob = []
    for ens_prob in ens_probs:
        final_prob = prob_add(final_prob, ens_prob)

    final_pred = [np.argsort(item)[-1:-4:-1] for item in final_prob]
    final_top_1, final_top_3 = 0, 0
    for pred, label in zip(final_pred, labels):
        if label in pred:
            final_top_3 += 1
        if label == pred[0]:
            final_top_1 += 1
    final_top_1 = final_top_1 * 100.0 / dataset_size
    final_top_3 = final_top_3 * 100.0 / dataset_size

    print("Finally Ensembled: top_1:{:.3f}\t top_3:{:0.3f}".format(final_top_1, final_top_3))
    time_spend = time.time() - time_start
    print("In %d min %1.3f sec." % (time_spend // 60, time_spend % 60))


if __name__ == '__main__':
    main()
