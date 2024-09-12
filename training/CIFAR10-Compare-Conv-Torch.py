import os
import copy
import random
import numpy as np
from tqdm import tqdm

from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load data and preprocess

path = os.path.join(".", "dataset", "cifar10")
os.makedirs(path, exist_ok=True)

# convert PIL image to tensor
transform = transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train = torchvision.datasets.CIFAR10(path, download=True, train=True, transform=transform)
test = torchvision.datasets.CIFAR10(path, download=True, train=False, transform=transform)

print(len(train), len(test))

def kaiming_init(m):
    """Applies Kaiming initialization to a given module."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
class CNN(nn.Module):

    def __init__(self, out_features: int, activation_type='selu', selu_dtype=torch.float32):
        super(CNN, self).__init__()
        self.selu_dtype = selu_dtype

        if activation_type == 'selu':
            activation = nn.SELU()
        elif activation_type == 'softplus':
            activation = nn.Softplus()
        elif activation_type == 'leakyrelu':
            activation = nn.LeakyReLU(0.3)
        else:
            activation = nn.ReLU()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),  # BatchNorm after Conv2d
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),  # BatchNorm after Conv2d
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),  # BatchNorm after Conv2d
            activation,
            nn.Flatten(),
            nn.Linear(in_features=128*4*4, out_features=2048),
            activation,
            nn.Linear(in_features=2048, out_features=1024),
            activation,
            nn.Linear(in_features=1024, out_features=out_features)
        )
        self.apply(kaiming_init)

        # for param in self.net.parameters():
        #     # biases zero
        #     if len(param.shape) == 1:
        #         nn.init.constant_(param, 0.0)
        #     # others using lecun-normal initialization
        #     else:
        #         nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        for layer in self.net:
            if isinstance(layer, nn.SELU) or isinstance(layer, nn.Softplus):
                x = layer(x.to(self.selu_dtype)).to(x.dtype)
            else:
                x = layer(x)
        return x

class Accuracy(nn.Module):

    def forward(self, x, y):

        y_pred = F.softmax(x, dim=1).argmax(dim=1).cpu().numpy()
        y = y.cpu().numpy()

        return accuracy_score(y_true=y, y_pred=y_pred)
def calculate_weight_max_difference(model1: nn.Module, model2: nn.Module) -> float:
    max_distance = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        distance = torch.max(torch.abs(param1.detach().view(-1) - param2.detach().view(-1))).item()
        if distance>max_distance:
            max_distance = distance
    return max_distance
def calculate_weight_difference(model1: nn.Module, model2: nn.Module) -> float:
    distance = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        distance += mean_squared_error(param1.detach().view(-1).cpu(), param2.detach().view(-1).cpu())
        # distance += wasserstein_distance(param1.detach().view(-1).cpu(), param2.detach().view(-1).cpu()) 
        # difference += torch.sum(torch.abs(param1.detach() - param2.detach()))
    return distance.item()

def _forward(network: nn.Module, data: DataLoader, metric: callable):

    for x, y in data:

        x = x.to(next(network.parameters()).device)

        y_hat = network.forward(x).cpu()
        loss = metric(y_hat, y)
        yield loss

@torch.enable_grad()
def update_with_weight_diff(network1: nn.Module, network2: nn.Module, data: DataLoader, 
                            loss: nn.Module, opt1: torch.optim.Optimizer, opt2: torch.optim.Optimizer) -> list:
    network1.train()
    network2.train()

    errs1 = []
    errs2 = []
    weight_diffs = []
    update_time = 0
    for x, y in data:
        x = x.to(next(network1.parameters()).device)
        y = y.to(next(network1.parameters()).device)

        # Forward pass for the first network (float32)
        y_hat1 = network1.forward(x)
        err1 = loss(y_hat1, y)
        errs1.append(err1.item())

        # Update first network (float32)
        opt1.zero_grad()
        err1.backward(retain_graph=True)
        opt1.step()

        # Forward pass for the second network (bfloat16)
        y_hat2 = network2.forward(x)
        err2 = loss(y_hat2, y)
        errs2.append(err2.item())

        # Update second network (bfloat16)
        opt2.zero_grad()
        err2.backward()
        opt2.step()
        update_time += 1

        # # Calculate weight differences
        # if (update_time % 20) == 0:
        #     # weight_diff = calculate_weight_max_difference(network1, network2)
            
        #     weight_diff = calculate_weight_difference(network1, network2)
        #     weight_diffs.append(weight_diff)
        #     print(f'weight diff: {weight_diff}')

    return errs1, errs2, weight_diffs

@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> float:

    network.eval()

    performance = []
    for p in _forward(network, data, metric):
        performance.append(p)
    return np.mean(performance).item()


def fit_with_weight_diff(network1: nn.Module, network2: nn.Module, trainloader: DataLoader, 
                         valloader: DataLoader, testloader: DataLoader, epochs: int, lr: float):
    optimizer1 = torch.optim.SGD(params=network1.parameters(), lr=lr)
    optimizer2 = torch.optim.SGD(params=network2.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    accuracy = Accuracy()

    train_losses, val_losses, accuracies, weight_diffs = [], [], [], []

    # performance before training
    val_losses.append(evaluate(network=network1, data=valloader, metric=ce))

    pbar = tqdm(range(epochs))
    for ep in pbar:
        # update networks
        tl1, tl2, wd = update_with_weight_diff(network1=network1, network2=network2, data=trainloader, 
                                         loss=ce, opt1=optimizer1, opt2=optimizer2)
        train_losses.extend(tl1)
        weight_diffs.extend(wd)

        vl = evaluate(network=network1, data=valloader, metric=ce)
        val_losses.append(vl)
        ac = evaluate(network=network1, data=valloader, metric=accuracy)

        accuracies.append(ac)

        print(f"train loss network1: {round(np.mean(tl1), 4):.4f}, "
              f"train loss network2: {round(np.mean(tl2), 4):.4f}, "
              f"val loss: {round(vl, 4):.4f}, "
              f"accuracy: {round(ac * 100, 2):.2f}%, "
              )

        pbar.set_description_str(desc=f"Epoch {ep+1}")

    acc = evaluate(network=best_model, data=testloader, metric=accuracy)
    print(f"Final accuracy on testset: {round(acc*100, 2):.2f}%")

    return train_losses, val_losses, accuracies, acc, weight_diffs
epochs = 40
lr = 3e-2
batch_size = 128
num_workers = 4

# obtain validation set (1/5 of train data to be equal to size of test data)
rng = np.random.default_rng(seed=42)
val_inds = rng.choice(np.arange(len(train)), size=len(train)//5, replace=False)
train_inds = np.delete(np.arange(len(train)), val_inds)

trainloader = DataLoader(Subset(train, indices=train_inds),
                         batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
valloader = DataLoader(Subset(train, indices=val_inds),
                       batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
testloader = DataLoader(test, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
# SELU training in bfloat16 and float32
network_float32 = CNN(out_features=10, activation_type = 'selu', selu_dtype=torch.float32).to(device)
network_bfloat16 = CNN(out_features=10, activation_type = 'selu', selu_dtype=torch.bfloat16).to(device)

# Copy weights from the first model to the second model
network_bfloat16.load_state_dict(network_float32.state_dict())

train_losses, val_losses, accuracies, acc, weight_diffs = fit_with_weight_diff(
    network_float32, network_bfloat16, trainloader, valloader, testloader, epochs, lr)
