# coding=utf-8
from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import argparse
import logging
from torchvision import transforms
import json
from PIL import Image
import pickle
import numpy as np
from time import time
from tqdm import tqdm

class_2_index = {
    '0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10,
    '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21,
    '29': 22,
    '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33,
    '4': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43, '49': 44,
    '5': 45, '50': 46, '51': 47, '52': 48, '53': 49, '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55,
    '6': 56, '60': 57, '61': 58, '62': 59, '63': 60, '64': 61, '65': 62, '66': 63, '67': 64, '68': 65, '69': 66,
    '7': 67, '70': 68, '71': 69, '72': 70, '73': 71, '74': 72, '75': 73, '76': 74, '77': 75, '78': 76, '79': 77,
    '8': 78, '9': 79}

index_2_class = {}
for k in class_2_index:
    v = class_2_index[k]
    index_2_class[v] = int(k)

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='Predict for test set')
parser.add_argument('--model_name', default='resnet50', type=str, help='Checkpoint path')
parser.add_argument('--ckpt_path', default='./resnet50/resnet50_places365_model_wts_1436.pth',
                    type=str, help='Checkpoint path')
parser.add_argument('--use_gpu', default=True, type=bool, help='Use gpu or cpu')
parser.add_argument('--val', default=False, type=bool, help='Validation or test')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--multi_scale', default='224,256,320,384', type=str, help='12 Crop with multi scale')
parser.add_argument('--crop_size', default=224, type=int, help='crop size')


args = parser.parse_args()
logging.info(args)

use_gpu = args.use_gpu
batch_size = args.batch_size
model_name = args.model_name
model_weights = args.ckpt_path

if args.val:
    testset_dir = './data/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
    result_path = './results/val.json'
    raw_result_path = './results/{}_raw_logits_val.json'.format(model_name)
else:
    testset_dir = '../data/test/scene_test_a_images_20170922'
    result_path = '../results/submit_{}.json'.format(model_name)
    raw_result_path = '../results/raw_logits_submit_{}.json'.format(model_name)


class SceneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filenames = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_model(model_weights):
    if use_gpu:
        model = torch.load(model_weights, pickle_module=pickle)
    else:
        model = torch.load(model_weights, map_location=lambda storage, loc: storage, pickle_module=pickle)

    for param in model.parameters():
        param.requires_grad = False
    print("Loading model from {}".format(model_weights))
    return model


def get_dataset(dataset_dir, transform):
    dataset_test = SceneDataset(root_dir=dataset_dir, transform=transform)
    return DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)


def predict(model, dataloader_test):
    model.eval()

    results = []
    for data in tqdm(dataloader_test):
        inputs, img_ids = data
        if use_gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs)

        outputs = model(inputs)

        result = [
            {"image_id": os.path.basename(img_id),
             "prob": prob.tolist()
             }
            for img_id, prob in zip(img_ids, outputs.data)
            ]

        results += result

    return results


def crop_transfroms(pos, img_size, crop_size):
    """
    12 Crop Evaluation
    :param i: between[0,11]
    :return: a transform
    """

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

    return transform


def prob_add(json1, json2):
    """
    add the prob from json files
    :param json1:
    :param json2:
    :return: json file with probability added.
    """
    if len(json1) == 0:
        return json2
    if len(json2) == 0:
        return json1

    new_prob = []
    for item1, item2 in zip(json1, json2):
        if item1['image_id'] != item2['image_id']:
            print("Warning: the img_id order in two json file are different.")
        else:
            img_id = item1['image_id']

        new_prob.append({
            "image_id": img_id,
            "prob": [p1 + p2 for p1, p2 in zip(item1['prob'], item2['prob'])]
        })
    return new_prob


def main():
    model = get_model(model_weights)

    ensemble_probs = []

    img_sizes = [int(s) for s in args.multi_scale]
    crop_size = args.crop_size

    for img_size in img_sizes:
        print("\nIMG_SIZE:%d" % img_size)
        ensemble_prob = []
        for i in range(12):
            print("\tCrop %d" % i)
            crop_transform = crop_transfroms(i, img_size, crop_size)
            dataloader_test = get_dataset(testset_dir, crop_transform)

            results = predict(model, dataloader_test)
            ensemble_prob = prob_add(ensemble_prob, results)
        ensemble_probs.append(ensemble_prob)

    print("Ensembling multiscale prediction..")
    final_predict = []
    for ensemble_prob in ensemble_probs:
        final_predict = prob_add(final_predict, ensemble_prob)

    results_json = []
    for item in final_predict:
        pred = np.argsort(item['prob'])[-1:-4:-1]
        results_json.append({
            'image_id': item['image_id'],
            'label_id': [index_2_class[l_id] for l_id in pred]
        })

    with open(result_path, 'w') as f:
        json.dump(results_json, f)
        print("Submit file saved in {}".format(result_path))

    with open(raw_result_path, 'w') as f:
        json.dump(final_predict, f)
        print("Submit file saved in {}".format(raw_result_path))


if __name__ == '__main__':
    main()