import flwr as fl


from strategies.fedavg import FedAvg
from strategies.qfedavg import QFedAvg
from strategies.qfedavg_manual_impl import QFedAvg_manual
from torch.nn import CrossEntropyLoss

if __name__ == "__main__":

    # Define strategy
    strategy = QFedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        
        #eval_fn = CrossEntropyLoss
    )
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
        config={"num_rounds": 3},
        strategy=strategy,
    )
