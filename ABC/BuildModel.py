import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# set device for training
device = "cuda"


# define a neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# create a instance of NeuralNetwork and move it to device
model = NeuralNetwork().to(device)
# print(model)

# build a random 28x28 matrix
X = torch.rand(1, 28, 28, device=device)
# prediction using model
logits = model(X)
pred_porbab = nn.Softmax(dim=1)(logits)
y_pred = pred_porbab.argmax(1)
print(f"Predicted class: {y_pred}")

# To break down the layers in model, let's take s images of size 28x28 as examples
input_image = torch.rand(3, 28, 28)
# first, flatten layer convert 2D 28*28 image into a 1D array
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
# then, linear layer applies linear transformation on input using stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
# activation functions help to introduce non-linearity in model
print(f"Before ReLU: {hidden1}")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
# nn.Sequential composes a bunch of modules into a whole model
seq_module = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
)
input_image = torch.rand(3, 28, 28)
logits = seq_module(input_image)
print(logits)
pred_porbab = nn.Softmax(dim=1)(logits)
print(pred_porbab)

# show model parameters
print("Model structure: ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
