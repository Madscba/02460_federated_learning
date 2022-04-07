import flwr as fl

from strategies.dp_fedavg import DPFedAvg
from strategies.fedavg import FedAvg
from strategies.qfedavg_fixed import QFedAvg
from global_model_eval import global_model_eval
from strategies.qfedavg_manual_impl import QFedAvg_manual
import argparse
import wandb
import os



FRACTION_FIT_ = 0.5
FRACTION_EVAL_ = 0.5
MIN_FIT_CLIENTS_ = 2
MIN_EVAL_CLIENTS_ = 2
MIN_AVAILABLE_CLIENTS_ = 2
test_file_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_test.txt'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select strategy')
    parser.add_argument("--strategy",type=str,default="FedAvg")
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--wandb_username', default=None)
    parser.add_argument('--wandb_mode', help='use "online" to log and sync with cloud', default='disabled')
    parser.add_argument('--configs', default='config.yaml')
    parser.add_argument('--rounds', default=200, type=int)
    parser.add_argument('--run_name', default='')
    parser.add_argument("--noise_multiplier",type=float,default=0.1)
    parser.add_argument("--noise_scale",type=float,default=1.0)
    parser.add_argument("--max_grad_norm",type=float,default=1.1)
    parser.add_argument("--target_delta",type=float,default=1e-5)
    parser.add_argument("--sample_rate",type=float,default=0.0025)
    parser.add_argument("--dataset_path",default='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist')
    args = parser.parse_args()

    if args.experiment_id:
        experiment="experiment-"+args.experiment_id
    else:
        experiment="experiment-"+wandb.util.generate_id()
    if args.wandb_username:
        os.environ['WANDB_USERNAME']=args.wandb_username

    config=os.path.join(os.getcwd(),'src','config',args.configs)
    wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
    wandb.init(project="02460_federated_learning", entity="02460-federated-learning", group=experiment, config=config, mode=args.wandb_mode,job_type='server')
    wandb.run.name = args.run_name+'_'+wandb.run.id
    wandb.config.update(args, allow_val_change=True)



    # Define strategy based on argument
    if args.strategy == "QFed_man":
        print("Strategy: Qfed_manual")
        strategy = QFedAvg_manual(
            q_param = 0.2,
            qffl_learning_rate = 0.01,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients = MIN_AVAILABLE_CLIENTS_)
    elif args.strategy == "QFed":
        print("Strategy: Qfed_flwr")
        strategy = QFedAvg(
            q_param = 0.2,
            qffl_learning_rate = 0.01,
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
            sample_rate=wandb.config.sample_rate,
            noise_multiplier=wandb.config.noise_multiplier,
            noise_scale=wandb.config.noise_scale,
            max_grad_norm=wandb.config.max_grad_norm,
            target_delta=wandb.config.target_delta
            )
    else:
        print("Strategy: FedAvg")
        strategy = FedAvg(eval_fn=global_model_eval,
            user_names_test_file=test_file_path,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_, 
            min_available_clients = MIN_AVAILABLE_CLIENTS_)

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )
