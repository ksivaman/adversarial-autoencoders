__author__ = 'Kirthi Shankar Sivamani'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from feature_losses import get_feature_maps
from train import Digits
import os

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 28, 28)
    return x

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset = MNIST('./data', download=True, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=False)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    lossNet = Digits()
    lossNet.load_state_dict(torch.load('models/mnist_cnn.pt'))

    for epoch in range(num_epochs):
        for data, ldata in zip(dataloader, train_loader):
            img, _ = data
            img = Variable(img).to(device)
            output = model(img)

            l_img, _ = ldata
            l_img = Variable(l_img).to(device)
            l_output = lossNet(l_img)

            #generate perceptual losses here --
            output_l3 = get_feature_maps(l_output, 3)
            img_l3 = get_feature_maps(l_img, 3)
            output_l7 = get_feature_maps(l_output, 7)
            img_l7 = get_feature_maps(l_img, 7)
            output_l10 = get_feature_maps(l_output, 10)
            img_l10 = get_feature_maps(l_img, 10)

            loss = criterion(output, img) + criterion(output_l3, img_l3) + criterion(output_l7, img_l7) + criterion(output_l10, img_l10)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))

    #torch.save(model.state_dict(), './models/conv_autoencoder.pth')

if __name__ == '__main__':
    main()
