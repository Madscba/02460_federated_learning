import os
import sys
sys.path.append("/work3/s173934/AdvML/02460_federated_learning/src/")
sys.path.append("/work3/s173934/AdvML/02460_federated_learning/src/data_exploration/")
from global_model_eval import global_model_eval
from data_exploration import visualize_category_distribution_over_classes 
import numpy as np
import time
from model import Net
import torch


if __name__ == "__main__":
    state_dict = "/work3/s173934/AdvML/02460_federated_learning/saved_models/Fedavg_state_dict_16_15_48.pt"
    data_folder = "/work3/s173934/AdvML/02460_federated_learning/dataset/test_stored_as_tensors"
    num_test_clients = 30 # i.e. fetch all clients
    get_loss = True

    client_names = os.listdir(data_folder)
    x_data = sorted([os.path.join(data_folder, client) for client in client_names if client.endswith("x.pt")])[:num_test_clients]
    y_data = sorted([os.path.join(data_folder, client) for client in client_names if client.endswith("y.pt")])[:num_test_clients]

    acc, loss, num_obs_per_user = global_model_eval(state_dict=state_dict,data_folder= data_folder, num_test_clients=num_test_clients, get_loss=get_loss)


    no_clients = 10
    min_clients = np.argsort(acc)[:no_clients] #Pick 10 clients with lowest accuracy
    max_clients = np.argsort(acc)[-no_clients:] #Pick 10 clients with highest accuracy

    visualize_category_distribution_over_classes([acc], title='Histogram of client accuracies (Validation)',labels=["Validation"],xlabel="Accuracy",ylabel="Amount of clients / Frequency")

    print("mean_acc:", np.mean(np.asarray(acc)))
    print(loss)
    print(num_obs_per_user)

