import wandb
import torch
import torchvision.transforms as transforms
from copy import deepcopy
from dp_sgd_utils import clip_gradients, add_noise
from FedOptLoss import FedOptLoss
#from privacy_opt import DP_SGD
from client_dataset import FemnistDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            wandb.log({"train_loss": loss.item()})
            optimizer.step()

def train_fed_prox(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = FedOptLoss(net.parameters(), mu=wandb.config.mu)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels, net.parameters())
            loss.backward()
            wandb.log({"train_loss": loss.item()})
            optimizer.step()

# def train_dp_sgd(net, trainloader, epochs):
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = DP_SGD(net, learning_rate=wandb.config.lr,momentum=wandb.config.momentum,
#                        sample_rate=wandb.config.sample_rate,max_grad_norm=wandb.config.max_grad_norm,
#                        noise_multiplier=wandb.config.noise_multiplier,target_delta=wandb.config.target_delta,
#                        lib=wandb.config.lib)
#     net.train()
#     theta0 = deepcopy(net.state_dict())
#     for _ in range(epochs):
#         for images, labels in trainloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             wandb.log({"train_loss": loss.item()})
#             optimizer.step()
#             if not optimizer.lib:
#                 clip_gradients(net=net,optimizer=optimizer,theta0=theta0,device=DEVICE)
#
#     if not optimizer.lib:
#         add_noise(net=net,optimizer=optimizer)
#
#     epsilon = optimizer.get_privacy_spent()
#     wandb.log({"epsilon": epsilon})

train_dict={'train': train, 'train_fed_prox': train_fed_prox, 'train_dp_sgd': None }#train_dp_sgd}

def choose_train_fn(train_fn='train'):
        return train_dict[train_fn]

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    wandb.log({"test_loss": loss, "test_accuracy": accuracy})
    return loss, accuracy


def load_data(user): 
    """Load Femnist (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    trainset = FemnistDataset(user, transform, train=True)
    testset = FemnistDataset(user, transform, train=False)
    trainloader = DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=wandb.config.batch_size)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples
