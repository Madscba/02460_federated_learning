import flwr as fl


from strategies.fedavg import FedAvg
from strategies.qfedavg import QFedAvg
from strategies.qfedavg_manual_impl import QFedAvg_manual
from torch.nn import CrossEntropyLoss
import argparse

FRACTION_FIT_ = 0.5
FRACTION_EVAL_ = 0.5
MIN_FIT_CLIENTS_ = 1
MIN_EVAL_CLIENTS_ = 1
MIN_AVAILABLE_CLIENTS_ = 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select strategy')
    parser.add_argument("--strategy",type=str,default="FedAvg")
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--wandb_username', default=None)
    args = parser.parse_args()
    if args.experiment_id:
        experiment="experiment-"+args.experiment_id
    else:
        experiment="experiment-"+wandb.util.generate_id()
    if args.wandb_username:
        os.environ['WANDB_USERNAME']=args.wandb_username

    wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
    wandb.init(project="02460_federated_learning", entity="02460-federated-learning", group=experiment, config=config, mode=args.wandb_mode,jobtype='server')
    wandb.run.name = args.user+wandb.run.id
    wandb.run.save()

    # Define strategy based on argument
    if args.strategy == "QFed_man":
        print("Strategy: Qfed_manual")
        strategy = QFedAvg_manual(
            q_param = 0.2,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients_ = MIN_AVAILABLE_CLIENTS_)
    elif args.strategy == "QFed":
        print("Strategy: Qfed_flwr")
        strategy = QFedAvg(
            q_param = 0.2,
            qffl_learning_rate = 0.1,
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_,
            min_available_clients_ = MIN_AVAILABLE_CLIENTS_)
    else:
        print("Strategy: FedAvg")
        strategy = FedAvg(
            fraction_fit=FRACTION_FIT_,
            fraction_eval=FRACTION_EVAL_,
            min_fit_clients=MIN_FIT_CLIENTS_,
            min_eval_clients=MIN_EVAL_CLIENTS_, 
            min_available_clients = MIN_AVAILABLE_CLIENTS_)
# uncomment this and outcomment what is above
###################################################################
    # # Define strategy
    # strategy = fl.server.strategy.QFedAvg(
    #     fraction_fit=0.5,
    #     fraction_eval=0.5,
    # )
# #################################################################

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 200},
        strategy=strategy,
    )
