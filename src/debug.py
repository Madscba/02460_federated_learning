import flwr as fl
from strategies.fedavg import FedAvg
from global_model_eval import global_model_eval
from model import Net
import wandb

dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'
wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
wandb.init(project="02460_federated_learning", entity="02460-federated-learning")
wandb.config.update({'dataset_path':dataset_path})
test_file_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_test.txt'
net=Net()
strategy = FedAvg(eval_fn=global_model_eval,user_names_test_file=test_file_path)
res=strategy.evaluate(net.parameters())