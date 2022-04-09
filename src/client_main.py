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
from main_utils import parse_args, update_config,sample_client,set_seed
import sys


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    #Set seed
    set_seed(args.seed)
    
    #Randomly sample a client
    user = sample_client(data_pool="Active")

    # Load model
    net = Net().to(DEVICE)
    
    if args.experiment_id:
        experiment="experiment-"+args.experiment_id
    else:
        experiment="experiment-"+wandb.util.generate_id()
    if args.wandb_username:
        os.environ['WANDB_USERNAME']=args.wandb_username

    
    config=os.path.join(os.getcwd(),'src','config',args.configs)
    wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
    wandb.init(project="02460_federated_learning", entity="02460-federated-learning", group=experiment, config=config, job_type=args.job_type, mode=args.wandb_mode)
    wandb.config.update(args, allow_val_change=True)
    wandb.run.name = user+'_'+wandb.run.id
    wandb.run.save()

    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(user)

    # Flower client
    qfed_client = args.qfed # default is false
    print("Running Qfed:",qfed_client)
    client=FemnistClient(net, trainloader, testloader, num_examples,
                         qfed_client=qfed_client,
                         train_fn=choose_train_fn(wandb.config.train_fn))

    # client.fit(Net().parameters(),config={'round':1})
    # Start client
    host = "localhost:8080" if sys.platform == "win32" else "[::]:8080" # needed for windows
    fl.client.start_numpy_client(host, client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args=parse_args(parser)
    main(args)

