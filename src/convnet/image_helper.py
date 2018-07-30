import configparser
import os
from shutil import copy

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

CONFIG = configparser.ConfigParser()
CONFIG.read('./src/config.ini')

def custom_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        img = Image.open(file)
        return img.convert(CONFIG['CNN Training']['image_mode'])

# Transforms are common image transforms. They can be chained together using Compose
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(42, 42)),
    transforms.ToTensor()
])

TRANSFORM_TRAINING = transforms.Compose([
    transforms.Resize(size=(42, 42)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

TRAIN_SET = torchvision.datasets.ImageFolder(CONFIG['CNN Image data']['training_data'],
                                             transform=TRANSFORM_TRAINING,
                                             loader=custom_pil_loader)
TEST_SET = torchvision.datasets.ImageFolder(CONFIG['CNN Image data']['test_data'],
                                            transform=TRANSFORM,
                                            loader=custom_pil_loader)
VALIDATION_SET = torchvision.datasets.ImageFolder(CONFIG['CNN Image data']['validation_data'],
                                                  transform=TRANSFORM_TRAINING,
                                                  loader=custom_pil_loader)

CLASSES = tuple(TRAIN_SET.classes)

# Training set
def train_set_loader():
    return torch.utils.data.DataLoader(TRAIN_SET, shuffle=True)

# Testing set
def test_set_loader():
    return torch.utils.data.DataLoader(TEST_SET, shuffle=False)

# Testing set
def validation_set_loader():
    return torch.utils.data.DataLoader(VALIDATION_SET, shuffle=False)

def save_image(path, img_folder):
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    copy(path, f"{img_folder}/{os.path.basename(path)}")
