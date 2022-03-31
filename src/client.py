    
from collections import OrderedDict
import flwr as fl
import torch
import wandb
from train_test_utils import test  


class FemnistClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, num_examples, run_qfed=False, train_fn=None) -> None:
        self.train=train_fn
        self.net=net
        self.num_examples=num_examples
        self.trainloader=trainloader
        self.testloader=testloader
        self.round=0
        self.run_qfed = run_qfed
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        # only return something meaningfull if self.qfed == true
        info = self.loss_prior_to_training()

        self.train(self.net, self.trainloader, epochs=wandb.config.epochs)
        wandb.log({"round": self.round})
        return self.get_parameters(), self.num_examples["trainset"], info

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    # only needed for q fed
    def loss_prior_to_training(self):
        info = {}
        if not self.run_qfed:
            return info # only return meaningfull value if we run qfed

        else:
            losses = []
            with torch.no_grad():
                loss_func = torch.nn.CrossEntropyLoss()
                for x, y in self.trainloader:
                    x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                    pred = self.net(x)
                    loss = loss_func(pred, y)
                    losses.append(loss)

            info["loss_prior_to_training"] = torch.mean(torch.stack(losses)).item()
            return info

