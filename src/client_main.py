import warnings
from client import FemnistClient
import flwr as fl
import torch
from model import Net
import argparse
import os
import wandb
from main_utils import load_data, parse_args, sample_client, set_seed, choose_model
import sys


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    #Set seed
    set_seed(args.seed)
    
    #Randomly sample a client
    user = sample_client(data_pool="Active")

    # Load model
    
    net = choose_model(args.model)()
    
    if args.experiment_id:
        experiment="experiment-"+args.experiment_id
    else:
        experiment="experiment-"+wandb.util.generate_id()
    if args.wandb_username:
        os.environ['WANDB_USERNAME']=args.wandb_username

    
    config=os.path.join(os.getcwd(),'src','config',args.configs)
    wandb.login(key=args.api_key)
    wandb.init(project="02460_federated_learning", entity=args.entity, group=experiment, config=config, job_type=args.job_type, mode=args.wandb_mode)
    wandb.config.update(args, allow_val_change=True)
    wandb.run.name = user+'_'+wandb.run.id
    wandb.run.save()


    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(user, args.num_classes)
    
    # Flower client
    qfed_client = args.qfed # default is false
    client=FemnistClient(net, trainloader, testloader, num_examples,
                         qfed_client=qfed_client)

    #client.fit(net.parameters(),config={'round':1})
    # Start client
    host = "localhost:8080" if sys.platform == "win32" else "[::]:8080" # needed for windows
    fl.client.start_numpy_client(host, client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args=parse_args(parser)
    main(args)

