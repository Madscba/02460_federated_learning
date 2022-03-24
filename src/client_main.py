from collections import OrderedDict
from mimetypes import init
import warnings
from client import FemnistClient
import flwr as fl
import torch
from model import Net
import argparse
import os
import wandb
from train_test_utils import choose_train_fn, load_data

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Load model
    net = Net().to(DEVICE)
    
    if args.experiment_id:
        experiment="experiment-"+args.experiment_id
    else:
        experiment="experiment-"+wandb.util.generate_id()
    
    config=os.path.join(os.getcwd(),'src','config',args.configs)
    wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
    wandb.init(project="02460_federated_learning", entity="02460-federated-learning", group=experiment, config=config,mode=args.wandb_mode)
    wandb.run.name = args.user+wandb.run.id
    wandb.run.save()

    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(args.user)

    # Flower client
    client=FemnistClient(net, trainloader, testloader, num_examples, choose_train_fn(wandb.config.train_fn), args)

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', 
                help='user ex f0000_14', default='f0000_14')
    parser.add_argument('--wandb_mode', 
                help='use "online" to log and sync with cloud', default='disabled')
    parser.add_argument('--configs', default='config.yaml')
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument("--dp_sgd",default=False)
    parser.add_argument("--opacus", default=False, help="Set true to use opacus library")
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
