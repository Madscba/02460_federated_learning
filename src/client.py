    
from collections import OrderedDict
import flwr as fl
import torch
import wandb
from train_test_utils import rank_pred, test, train  


class FemnistClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, num_examples, qfed_client=False, user = None) -> None:
        self.net=net
        self.num_examples=num_examples
        self.trainloader=trainloader
        self.testloader=testloader
        self.qfed_client = qfed_client
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.user = user
        super().__init__()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # only return something meaningfull if self.qfed == true
        info = self.loss_prior_to_training()

        train_loss=train(self.net,
                         self.trainloader,
                         round=config['round'],
                         epochs=wandb.config.epochs,
                         lr=wandb.config.lr)

        info['loss']=train_loss
        info['user'] = self.user
        return self.get_parameters(), self.num_examples["trainset"], info

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy,ranked_pred = test(self.net, self.testloader,config['round'])
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy),"ranked_pred":ranked_pred}

    # only needed for q fed
    def loss_prior_to_training(self):
        info = {}
        if not self.qfed_client:
            return info # only return meaningfull value if we run qfed

        else:
            with torch.no_grad():
                # just make a forward pass with all of the data to optimize speed
                # this could instead be done with if self.trainloader.__len__() < 100.
                try:
                    x, y = [], []
                    for x_, y_ in self.trainloader:
                        x.append(x_.to(self.DEVICE))
                        y.append(y_.to(self.DEVICE))
                    x = torch.cat(x)
                    y = torch.cat(y)
                    pred = self.net(x)
                    loss = self.loss_func(pred, y).item()
                    info["loss_prior_to_training"] = loss

                # for some reason we couldnt pass all of the data through in one batch most liekly due to
                # memory limitations.
                except RuntimeError:
                    print("exception happened, prolly memory error in client.py line 63")
                    losses = []
                    for x, y in self.trainloader:
                        x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                        pred = self.net(x)
                        loss = self.loss_func(pred, y)
                        losses.append(loss)

                    info["loss_prior_to_training"] = torch.mean(torch.stack(losses)).item()

            return info

