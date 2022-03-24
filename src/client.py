    
from collections import OrderedDict
import flwr as fl
import torch
import wandb
from opacus import PrivacyEngine
from privacy_opt import DP_SGD
from train_test_utils import test  


class FemnistClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, num_examples, train_fn=None, args=None) -> None:
        self.train=train_fn
        self.net=net
        self.num_examples=num_examples
        self.trainloader=trainloader
        self.testloader=testloader
        self.round=0
        self.dp_sgd = args.dp_sgd
        self.opacus = args.opacus
        if self.dp_sgd:
            self.noise_multiplier = args.noise_multiplier
            self.max_grad_norm = args.max_grad_norm
            self.target_delta = args.target_delta
            self.sample_rate = args.sample_rate
            if self.opacus:
                self.privacy_engine = PrivacyEngine(
                    self.net,
                    sample_rate=self.sample_rate,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                    noise_multiplier=self.noise_multiplier,
                    accountant='gdp'
                )
            else:
                self.privacy_engine = DP_SGD(
                    sample_rate=self.sample_rate,
                    max_grad_norm=self.max_grad_norm,
                    noise_multiplier=self.noise_multiplier,
                    target_delta=self.target_delta
                )
        super().__init__()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.round+=1
        if self.dp_sgd:
            if self.opacus:
                self.train(net=self.net, trainloader=self.trainloader,
                             privacy_engine=self.privacy_engine,
                             epochs=wandb.config.epochs, target_delta=self.target_delta)
            else:
                self.train(net=self.net, trainloader=self.trainloader,
                             privacy_engine=self.privacy_engine,
                             epochs=wandb.config.epochs, target_delta=self.target_delta,
                             noise_multiplier=self.noise_multiplier,
                             max_grad_norm=self.max_grad_norm)
        else:
            self.train(self.net, self.trainloader, epochs=wandb.config.epochs)
        wandb.log({"round": self.round})
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}