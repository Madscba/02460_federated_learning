import wandb
import torch
import torchvision.transforms as transforms
from copy import deepcopy
from dp_sgd_utils import clip_gradients, add_noise
from FedOptLoss import FedOptLoss


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader,round, epochs):
    """Train the network on the training set."""
    criterion = configure_criterion(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    loss_agg=0
    theta0 = deepcopy(net.state_dict())
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if wandb.config.strategy in ['FedProx']:
                loss = criterion(net(images), labels, net.parameters())
            else:
                loss = criterion(net(images), labels)
            loss.backward()
            loss_agg+=loss.item()
            wandb.log({"train_loss": loss.item()})
            optimizer.step()
            if wandb.config.strategy in ['DP_Fed', 'FedX']:
                clip_gradients(net=net,max_grad_norm=wandb.config.max_grad_norm,theta0=theta0,device=DEVICE)
    avg_train_loss=loss_agg/(len(trainloader)*epochs)
    wandb.log({'round': round,"train_loss_round": avg_train_loss})
    return avg_train_loss

def configure_criterion(parameters):
    if wandb.config.strategy in ['FedProx']:
        criterion= FedOptLoss(parameters, mu=wandb.config.mu)
    else: 
        criterion=torch.nn.CrossEntropyLoss()
    return criterion

train_dict={'train': train}

def choose_train_fn(train_fn='train'):
        return train_dict[train_fn]

def test(net, testloader,round):
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
    wandb.log({"round":round,"test_loss": loss, "test_accuracy": accuracy})
    return loss, accuracy

