import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zores(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# Lambda transforms apply any user-defined lambda function.
# torch.scatter_() takes elements from src to output according to index matrix
