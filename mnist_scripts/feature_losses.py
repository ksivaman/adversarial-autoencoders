from train import Digits
import torch

model = Digits()
model.load_state_dict(torch.load('models/mnist_cnn.pt'))

print(model)
