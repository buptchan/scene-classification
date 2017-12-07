# coding=utf-8
from __future__ import print_function
import os

import logging
import argparse
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from logger import Logger
from utils import *
from dataset import SceneDataset
from models.inception_v4 import InceptionV4
from models.se_module import se_resnet50


logging.basicConfig(level=logging.INFO)


def cross_entropy_loss(outputs, targets):
    loss = -torch.mean(torch.sum(torch.log(outputs) * targets, 1))
    return loss


def train_epoch(model, dataset, optimizer, logger=None, criterion=cross_entropy_loss):
    model.train(True)
    dataset.train()

    dataloader = dataset.get_dataloader()

    step = 0
    running_loss = 0.0
    top_1 = AverageMeter()
    top_3 = AverageMeter()
    iter_size = 8

    for data in dataloader:
        inputs, labels = data

        '''  Label Smoothing   https://arxiv.org/abs/1512.00567  '''
        epsilon = 0.05
        K = 80
        onehot_labels = torch.ones(labels.shape[0], 80) * epsilon / K
        for onehot_label, label in zip(onehot_labels, labels):
            onehot_label[label] += 1 - epsilon

        inputs, onehot_labels, labels = inputs.cuda(), onehot_labels.cuda(), labels.cuda()
        inputs, onehot_labels, labels = Variable(inputs), Variable(onehot_labels), Variable(labels)

        outputs = model(inputs)

        softmax = torch.nn.Softmax()
        loss = criterion(softmax(outputs), onehot_labels) / iter_size

        loss.backward()
        '''累积16个mini batch的平均梯度进行反传'''
        if step % iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.data[0] * iter_size

        acc_1, acc_3 = accuracy(outputs.data, labels.data, top_k=(1, 3))
        acc_1, acc_3 = acc_1[0], acc_3[0]
        top_1.update(acc_1, inputs.data.size(0))
        top_3.update(acc_3, inputs.data.size(0))

        if step % 100 == 0:
            print("Step:{}\t Loss:{:1.3f}\ttop_1:{:1.3f}\ttop_3:{:1.3f}".format(
                step, loss.data[0] * iter_size, acc_1, acc_3))

        if step % 100 == 0:
            if logger:
                info = {
                    'loss': loss.data[0] * iter_size,
                    'accuracy/top_1': acc_1,
                    'accuracy/top_3': acc_3,
                    'learning_rate': cur_lr
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, global_step + step)

        step += 1

    epoch_loss = running_loss * batch_size / len(dataset)

    return epoch_loss, top_1.avg, top_3.avg


def val_epoch(model, dataset, logger=None, criterion=nn.CrossEntropyLoss()):
    model.eval()
    dataset.eval()

    dataloader = dataset.get_dataloader()
    print("Validating...")

    running_loss = 0.0
    top_1 = AverageMeter()
    top_3 = AverageMeter()

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]

        acc_1, acc_3 = accuracy(outputs.data, labels.data, top_k=(1, 3))
        acc_1, acc_3 = acc_1[0], acc_3[0]
        top_1.update(acc_1, inputs.data.size(0))
        top_3.update(acc_3, inputs.data.size(0))

    epoch_loss = running_loss * val_batch_size / len(dataset)

    print("Val loss:{:1.3f},\ttop_1:{:1.3f},\ttop_3:{:1.3f}".format(
        epoch_loss, top_1.avg, top_3.avg))

    if logger:
        val_info = {
            'loss': epoch_loss,
            'accuracy/top_1': top_1.avg,
            'accuracy/top_3': top_3.avg,
            'learning_rate': cur_lr
        }
        log_step = global_step + len(dataset) / batch_size
        for tag, value in val_info.items():
            logger.scalar_summary(tag, value, log_step)
    else:
        print("Warning:No logger for val!")

    return epoch_loss, top_1.avg, top_3.avg


def get_trainable_variables(model, retrain):
    if not retrain:
        for param in model.parameters():
            param.requires_grad = False  # Save memory on GPU
        trainable_scope = [model.fc]
    else:
        for param in model.parameters():
            param.requires_grad = False
        # Modify the trainable scope as you like. e.g.:[model.fc, model.layer4, model.layer3]
        trainable_scope = [model]

    trainable_variables = []
    for scope in trainable_scope:
        for param in scope.parameters():
            param.requires_grad = True
            trainable_variables.append(param)
    return trainable_variables


def adjust_lr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    print("Adjust Learning Rate By {:.2f}".format(decay_rate))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model_name', default='resnet50', type=str, help='Model name')
    parser.add_argument('--optim', default='SGD', type=str, help='Optimizer, SGD/Adam')
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='learning rate,default is 3e-4')
    parser.add_argument('--checkpoint_path', default='./resnet50/resnet50_places365_model_wts_367984.pth',
                        type=str, help='ckpt path')
    parser.add_argument('--train_dir', default='./resnet50', type=str, help='ckpt path')
    parser.add_argument('--log_dir', default='./logs/resnet50', type=str, help='log path')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='learning rate decay')
    parser.add_argument('--decay_milestones', default='10,15', type=str, help='lr decay policy')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size default is 64')
    parser.add_argument('--global_step', default=0, type=int, help='Global step of the model')
    parser.add_argument('--img_size', default=224, type=int, help='image size')
    parser.add_argument('--device', default='0', type=str, help='assign a GPU device')
    parser.add_argument('--retrain', default=False, type=bool, help='Train from a fine tuned model')
    parser.add_argument('--image_enhance', default=False, type=bool, help='Use image enhance or not')
    args = parser.parse_args()
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    arch = args.model_name
    learning_rate = args.learning_rate
    decay_milestones = [int(m) for m in args.decay_milestones.split(',')]
    global_step = args.global_step
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    retrain = args.retrain
    val_batch_size = 4
    train_dir = args.train_dir
    log_dir = args.log_dir
    ending_lr = 1e-5  # The minimal learning rate during training
    cur_lr = learning_rate

    best_acc = 0
    t0 = time.time()

    global global_step, batch_size, val_batch_size
    use_gpu = torch.cuda.is_available()

    if retrain:
        model_weight = checkpoint_path
    else:
        model_weight = './models/pretrained_model/%s_places365.pth.tar' % arch

    # Set the logger
    logger = Logger(log_dir)
    val_logger = Logger(os.path.join(log_dir, 'val'))

    print('==> Building Model')
    model = get_model(model_weight, use_gpu)

    trainable_variables = get_trainable_variables(model, args.retrain)

    # Get the optimizer. I just used 2 optimizers: Adam and SGD with momentum.
    optimizer = get_optimizer(trainable_variables, learning_rate=args.learning_rate, optim_name=args.optim)

    for epoch in range(args.num_epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        global cur_lr

        if epoch in decay_milestones:
            if cur_lr > ending_lr:
                adjust_lr(optimizer, args.decay_rate)

        # Sampling the training set with "label shuffling"
        dataset = SceneDataset(batch_size=args.batch_size,
                               val_batch_size=val_batch_size,
                               img_size=args.img_size,
                               image_enhance=args.image_enhance)

        train_loss, train_acc_1, train_acc_3 = train_epoch(model, dataset, optimizer, logger=logger)
        print("Training loss:{:1.3f}\ttop_1:{:1.3f}\ttop_3:{:1.3f}".format(train_loss, train_acc_1, train_acc_3))

        val_loss, val_acc_1, val_acc_3 = val_epoch(model, dataset, logger=val_logger)

        save_path = os.path.join(train_dir, '%s_model_wts_%d.pth' % (arch, global_step))
        if val_acc_3 > best_acc:
            best_acc = val_acc_3 - 0.05
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            torch.save(model, save_path)
            print("Model saved in %s" % save_path)

        global_step += (80 * 862) / args.batch_size

        duration = time.time() - t0
        t0 = time.time()
        print("Epoch {}:\tin {} min {:1.2f} sec".format(epoch, duration // 60, duration % 60))

if __name__ == '__main__':
    main()
