from email.policy import default
import wandb
import sys
import wandb
from client_dataset import FemnistDataset
from torchvision import transforms
from torch.utils.data import DataLoader 
import random
import numpy as np
import torch
from model import mlr, Net

def load_data(user,num_classes): 
    """Load Femnist (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = FemnistDataset(user, transform, train=True,num_classes=num_classes)
    testset = FemnistDataset(user, transform, train=False,num_classes=num_classes)
    trainloader = DataLoader(trainset, batch_size=wandb.config.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=wandb.config.batch_size)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples



def parse_args(parser):
    #parser.add_argument('--user', 
    #            help='user ex f0000_14', default='f0000_14')
    parser.add_argument('--wandb_mode', 
                help='use "online" to log and sync with cloud', default='disabled')
    parser.add_argument('--configs', default='config.yaml')
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--wandb_username', default=None)
    parser.add_argument('--job_type', default='client')
    parser.add_argument('--dataset_path',default=None)
    parser.add_argument('--num_classes',default=None,type=int)
    parser.add_argument('--seed', type=int,help='set integer seed', default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.90
    )
    parser.add_argument(
        "--qfed",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--straggler",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--model",
        default='Net'
    )
    parser.add_argument(
        "--api_key",
        default=None
    )
    parser.add_argument(
        "--entity",
        default=None
    )
    parser.add_argument("--wandb_project", default='02460_federated_learning', type=str)
    args = parser.parse_args()
    return args


def sample_client(data_pool="Active"):
    """
    returns a sample from specified data_pool (Active: 2996 user) (Test: 596 users)
    """

    if data_pool == "Active":
        usernames = open('src/data/usernames_test.txt').read().splitlines()
    else:
        usernames = open('src/data/usernames_test.txt').read().splitlines()

    username = random.choice(usernames)
    return username

def set_seed(seed):
    """
    Seed sampling of clients, random noise, etc.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def choose_model(model_name):
    model_dict={'Net': Net, 'mlr': mlr}
    return model_dict[model_name]
