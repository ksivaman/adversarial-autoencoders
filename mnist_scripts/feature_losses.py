from train import Digits, test

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision import datasets

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

my_vgg = vgg16(pretrained=True)
my_vgg.to(device)

def get_feature_maps(input_image, layer_from_back=3):
    return my_vgg.features[:-layer_from_back](input_image)

model = Digits()
model.load_state_dict(torch.load('models/mnist_cnn.pt'))

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True,
                    transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=128, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=1000, shuffle=True, **kwargs)

model.to(device)
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = get_feature_maps(data, 3)
        output2 = get_feature_maps(data, 7)
        output3 = get_feature_maps(data, 10)
        print(output.shape, output2.shape, output3.shape)
