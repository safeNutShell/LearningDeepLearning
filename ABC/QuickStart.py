import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Download training data from FashionMNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# Download test data from same dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
# Create data loaders which can warp an iterable over dataset
train_dataLoader = DataLoader(training_data, batch_size=batch_size)
test_dataLoader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataLoader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    # see only one sample
    break
# see more on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# create a simple model
device = "cuda"  # RTX 3060!!!


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# hint: CUDA capability of GPU should match with PyTorch
model = NeuralNetwork().to(device)
print(model)

# see more on https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# use loss function and optimizer to train the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batchse = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batchse
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------")
    train(train_dataLoader, model, loss_fn, optimizer)
    test(test_dataLoader, model, loss_fn)
print("Done!")

# save this model
torch.save(model.state_dict(), "test_model.pth")
print("Save model state to test_model.pth")

model2 = NeuralNetwork()
model2.load_state_dict(torch.load("test_model.pth"))
print(model2)
