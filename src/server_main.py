from email.policy import default
import flwr as fl

from strategies.dp_fedavg import DPFedAvg
from strategies.fedavg import FedAvg
from strategies.qfedavg_fixed import QFedAvg
from strategies.fedx import FedX
from global_model_eval import global_model_eval
from strategies.qfedavg_manual_impl import QFedAvg_manual
import argparse
import wandb
import os
from flwr.server.client_manager import SimpleClientManager
from server import ServerDisconnect
from main_utils import choose_model



FRACTION_FIT_ = 0.5
FRACTION_EVAL_ = 0.5
MIN_FIT_CLIENTS_ = 10
MIN_EVAL_CLIENTS_ = 2
MIN_AVAILABLE_CLIENTS_ = 10
test_file_path='/work3/s173934/AdvML/02460_federated_learning/dataset/test_stored_as_tensors'
#test_file_path="C:/Users/Karlu/Desktop/advanced/02460_federated_learning/dataset/test_stored_as_tensors"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select strategy')
    parser.add_argument("--strategy",type=str,default="Fed_avg")
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--wandb_username', default=None)
    parser.add_argument('--wandb_mode', help='use "online" to log and sync with cloud', default='disabled')
    parser.add_argument('--configs', default='config.yaml')
    parser.add_argument('--rounds', default=2, type=int)
    parser.add_argument('--run_name', default='')
    parser.add_argument("--noise_multiplier",type=float,default=0.1)
    parser.add_argument("--noise_scale",type=float,default=None)
    parser.add_argument("--max_grad_norm",type=float,default=1.1)
    parser.add_argument("--target_delta",type=float,default=1e-5)
    parser.add_argument("--sample_rate",type=float,default=0.0025)
    parser.add_argument("--q_param",type=float,default=0.2)
    parser.add_argument("--dataset_path",default='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_num_clients", type=int, default=1000)
    parser.add_argument("--model", default='Net', type=str)
    parser.add_argument("--wandb_project", default='02460_federated_learning', type=str)
    parser.add_argument(
        "--api_key",
        default=None
    )
    parser.add_argument(
        "--entity",
        default=None
    )
    args = parser.parse_args()

    if args.experiment_id:
        experiment="experiment-"+args.experiment_id
    else:
        experiment="experiment-"+wandb.util.generate_id()
    if args.wandb_username:
        os.environ['WANDB_USERNAME']=args.wandb_username

       
    config=os.path.join(os.getcwd(),'src','config',args.configs)
    wandb.login(key=args.api_key)
    wandb.init(project=args.wandb_project, entity=args.entity, group=experiment, config=config, mode=args.wandb_mode,job_type='server')
    wandb.run.name = args.run_name+'_'+wandb.run.id
    wandb.config.update(args, allow_val_change=True)



    # Define strategy based on argument
    if args.strategy == "Qfed_manual":
        print("Strategy: Qfed_manual")
        strategy = QFedAvg_manual(
            q_param = wandb.config.q_param,
            qffl_learning_rate = wandb.config.lr,
            eval_fn=global_model_eval,
            test_file_path=test_file_path,
            num_rounds=args.rounds,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients = MIN_AVAILABLE_CLIENTS_)
    elif args.strategy == "Qfed_flwr":
        print("Strategy: Qfed_flwr_fixed")
        strategy = QFedAvg(
            q_param = 0.2,
            qffl_learning_rate = 0.001,
            num_rounds=args.rounds,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients = MIN_AVAILABLE_CLIENTS_)
    elif args.strategy == "DP_Fed":
        print("Strategy: DP_FedAvg")
        strategy = DPFedAvg(
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients=MIN_AVAILABLE_CLIENTS_,
            num_rounds=args.rounds,
            batch_size=wandb.config.batch_size,
            noise_multiplier=wandb.config.noise_multiplier,
            noise_scale=wandb.config.noise_scale,
            max_grad_norm=wandb.config.max_grad_norm,
            total_num_clients=wandb.config.total_num_clients
            )
    elif args.strategy == "FedX":
        print("Strategy: FedX")
        strategy = FedX(
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients=MIN_AVAILABLE_CLIENTS_,
            num_rounds=args.rounds,
            batch_size=wandb.config.batch_size,
            noise_multiplier=wandb.config.noise_multiplier,
            noise_scale=wandb.config.noise_scale,
            max_grad_norm=wandb.config.max_grad_norm,
            total_num_clients=wandb.config.total_num_clients,
            q_param = wandb.config.q_param,
            qffl_learning_rate = wandb.config.lr
            )
    else:
        print("Strategy: FedAvg")
        strategy = FedAvg(
            model=choose_model(args.model),
            eval_fn=global_model_eval,
            test_file_path=test_file_path,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            num_rounds=args.rounds,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_, 
            min_available_clients = MIN_AVAILABLE_CLIENTS_)

    server=ServerDisconnect(client_manager=SimpleClientManager(),strategy=strategy)
    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": args.rounds},
        strategy=strategy,server=server)
    
