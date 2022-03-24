from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from FedOptLoss import FedOptLoss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from opacus import PrivacyEngine
from copy import deepcopy
from model import Net
from client_dataset import FemnistDataset
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, privacy_engine, epochs, target_delta, noise_multiplier, max_grad_norm):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    theta0 = deepcopy(net.state_dict())
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            # clip gradients for each layer
            for layer, params in net.state_dict().items():
                diff = params-theta0[layer]
                params.data = theta0[layer]+per_layer_clip(diff,max_grad_norm)

    # Add noise to model parameters
    for param in net.parameters():
        param.data += torch.normal(mean=0, std=noise_multiplier)
    epsilon, _ = privacy_engine.accountant.get_privacy_spent(delta=target_delta)
    return epsilon

def per_layer_clip(layer_gradient,max_grad_norm):
    total_norm = torch.norm(layer_gradient.data.detach(),p=1).to(DEVICE)
    clipped_gradient = layer_gradient*min(1,max_grad_norm/total_norm)
    return clipped_gradient

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
    return loss, accuracy


def load_data(user, root_dir):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    trainset = FemnistDataset(user, root_dir, transform, train=True)
    testset = FemnistDataset(user, root_dir, transform, train=False)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main(args):
    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(args.user, args.dataset_root)

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self, net, trainloader, testloader, args):
            self.net = net
            self.trainloader = trainloader
            self.testloader = testloader
            self.dpsgd = args.dpsgd
            if args.dpsgd:
                self.noise_multiplier = args.noise_multiplier
                self.max_grad_norm = args.max_grad_norm
                self.target_delta = args.target_delta
                self.sample_rate = args.sample_rate
                self.privacy_engine = PrivacyEngine(
                    self.net,
                    sample_rate=self.sample_rate,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                    noise_multiplier=self.noise_multiplier,
                    accountant='gdp'
                )

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            epsilon = train(net=self.net, trainloader=self.trainloader, privacy_engine=self.privacy_engine,
                            epochs=1,target_delta=self.target_delta,noise_multiplier=self.noise_multiplier,
                            max_grad_norm=self.max_grad_norm)
            print(f"epsilon = {epsilon:.2f}")
            return self.get_parameters(), num_examples["trainset"], {'epsilon': epsilon}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                    help='root to data ex. data/femnist')
    parser.add_argument('--user', 
                help='user ex f0000_14')
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max_grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.01,
        metavar="Q",
        help="Sample rate (determined on server level)",
    )
    args = parser.parse_args()
    main(args)