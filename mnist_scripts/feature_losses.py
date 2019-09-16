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

my_vgg = vgg16(pretrained=True)

def get_features_L3(input_image):
    return my_vgg.features[:-3](input_image)

model = Digits()
model.load_state_dict(torch.load('models/mnist_cnn.pt'))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=128, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        # transforms.Grayscale(num_output_channels=3),
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
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_accuracy = correct / (10 * len(test_loader))

test_loss /= len(test_loader.dataset)
print('Test accuracy: {}'.format(test_accuracy))

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = get_features_L3(data)
        print(output.shape)

