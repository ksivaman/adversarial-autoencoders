import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from autoencoder import Autoencoder

import sys
sys.path.append('../')

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_dataset

'''
before running in the script run the following in the script directory: 
> mkdir tensors
> mkdir models
'''

#Create the neural network architecture, return logits instead of activation in forward method (Eg. softmax).
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)

#set cuda
device = torch.device('cuda')

# Obtain the model object
model = Net()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Initialize the classifier
cifar_classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=criterion, optimizer=optimizer, 
                                     input_shape=(1, 28, 28), nb_classes=10)

# Train the classifier
cifar_classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)

# Test the classifier
predictions = cifar_classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy before attack: {}%'.format(accuracy * 100))

# Craft the adversarial examples
epsilon = 0.2  # Maximum perturbation
adv_crafter = FastGradientMethod(cifar_classifier, eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test)

# Test the classifier on adversarial exmaples
predictions = cifar_classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy after attack: {}%'.format(accuracy * 100))

cifar_classifier.save('cifar_fgsm_state_dict', 'models')

preprocess = Autoencoder()
preprocess.load_state_dict(torch.load('models/conv_autoencoder.pth'))

x_test_adv = torch.from_numpy(x_test_adv)
x_test_denoised = preprocess(x_test_adv)
x_test_denoised = x_test_denoised.detach().numpy()

predictions = cifar_classifier.predict(x_test_denoised)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy after defense: {}%'.format(accuracy * 100))
