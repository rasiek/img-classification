
#imports

import imp
import os
import torch
from torch.utils import data
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import dataset, random_split
from tensorflow import keras

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import matplotlib
import matplotlib.pyplot as plt

#Download dataset

#dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"

# download_url(dataset_url, ".")

# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#     tar.extractall(path='./data')


data_dir = './data/cifar10'

# #Dataset Testing
# airplane_files = os.listdir(data_dir + "/train/airplane")
# print("No. airplane exes: ", len(airplane_files))
# print(airplane_files[:5])

ds = ImageFolder(data_dir + '/train', transform=ToTensor())

print(len(ds))

# # Image Tensor
# img, label = ds[0]
# print(img.shape, label)
# print(img)


# #Example show
# matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# def show_example(img, label):
#     print(f'Label: {ds.classes[label]}({label})')
#     plt.imshow(img.permute(1,2,0))
#     plt.show()


# show_example(*ds[0])


random_seed = 42
torch.manual_seed(random_seed)

val_size = 5000
train_size = len(ds) -val_size

train_ds, val_ds = random_split(ds, [train_size, val_size])

print(len(train_ds), len(val_ds))



