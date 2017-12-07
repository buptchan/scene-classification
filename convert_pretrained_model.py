import torch
import torch.nn as nn
import torchvision.models as models

from inception_v4 import InceptionV4
from se_module import se_resnet50

resnet_18_path = './pretrained/whole_resnet18_places365.pth.tar'
resnet_50_path = './pretrained/whole_resnet50_places365.pth.tar'
densenet_161_path = './pretrained/whole_densenet161_places365.pth.tar'
inception_v4_path = './pretrained/inceptionv4-97ef9c30.pth'


def convert_resnet18(path=resnet_18_path):
    model = torch.load(path)
    model.fc = nn.Linear(512, 80)
    torch.save(model, './pretrained/resnet18_place365.pth.tar')
    return model


def convert_resnet50(path=resnet_50_path):
    model = torch.load(path)
    model.fc = nn.Linear(2048, 80)
    torch.save(model, './pretrained/resnet50_place365.pth.tar')
    return model


def convert_densenet161(path=densenet_161_path):
    model = torch.load(path)
    model.classifier = nn.Linear(2048, 80)
    torch.save(model, './pretrained/densenet161_places365.pth.tar')
    return model


def convert_se_resnet50(path=resnet_50_path):
    """
    Load weights in ResNet50 to SE-ResNet50
    Also, you can convert Inception model into the SE-Version in this way.
    :param path: ResNet model path
    :return: SE-ResNet model
    """
    model = se_resnet50(num_classes=365)
    resnet50 = torch.load(path)
    state_dict = model.state_dict()
    weights = resnet50.state_dict()
    for k in state_dict:
        if k in weights:
            state_dict[k] = weights[k]
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(2048, 80)
    torch.save(model, './pretrained/se_resnet50.pth')
    return model


def convert_inception_v4(path=inception_v4_path):
    weights = torch.load(path)
    model = InceptionV4()
    state_dict = model.state_dict()
    for k in state_dict:
        if k in weights:
            state_dict[k] = weights[k]
    model.load_state_dict(state_dict)
    torch.save(model, './pretrained/inception_v4_imageNet.pth')
    return model