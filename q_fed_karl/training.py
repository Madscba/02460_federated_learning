from cifar10_noniid import get_dataset_cifar10_extr_noniid
from network import cifar_net
import torch.optim as optim
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch


# data
num_clients = 100
nclass = 4
nsamples = 100
total_samples_client = int(nclass*nsamples)

rate_unbalance = 1.0
train_dataset, test_dataset, client_groups_train, client_groups_test = \
    get_dataset_cifar10_extr_noniid(num_clients, nclass, nsamples, rate_unbalance)
train_dataset.data = train_dataset.data/255
train_dataset.targets = np.asarray(train_dataset.targets).astype(int)

test_dataset.data = test_dataset.data/255
test_dataset.targets = np.asarray(test_dataset.targets).astype(int)



# network and optimizer
model_server = cifar_net()
model_server.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_server.parameters(), lr=0.001, momentum=0.9)


# training hyper params:
server_epochs = 10
batch_size = 20 # should have "total_samples_client % batch_size = 0"
num_client_subset = 10 # number of clients to learn from before updating the main model
client_epochs = 2 # number of epoch to run per client


for server_epoch in range(server_epochs):
    training_order = np.arange(num_clients)
    np.random.shuffle(training_order)

    client_model_params = []
    client_subset_loss = 0
    for client_idx, client in enumerate(training_order):
        print(client_idx)
        model_client = deepcopy(model_server)
        optimizer = optim.SGD(model_client.parameters(), lr=0.1, momentum=0.8)

        for _ in range(client_epochs):
            # extract the clients data and convert to tensor on the GPU
            client_data_x = torch.from_numpy(train_dataset.data[client_groups_train[client].astype(int)]).type(torch.FloatTensor).cuda()
            client_data_y = torch.from_numpy(train_dataset.targets[client_groups_train[client].astype(int)]).type(torch.long).cuda()

            # shuffle the clients data:
            random_perm = torch.randperm(total_samples_client)
            client_data_x = client_data_x[random_perm]
            client_data_y = client_data_y[random_perm]

            for batch_idx in range(total_samples_client // batch_size):
                optimizer.zero_grad()
                batch_x = torch.reshape(client_data_x[batch_idx*batch_size:(batch_idx+1)*batch_size], (batch_size, 3,32,32))
                batch_y = client_data_y[batch_idx*batch_size:(batch_idx+1)*batch_size]
                pred = model_client.forward(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        client_subset_loss += loss.detach().cpu()



        client_model_params.append(deepcopy(list(model_client.parameters())))

        # update server model every num_client_subset
        if (1+client_idx) % num_client_subset == 0:
            # make sure to not track the gradient
            with torch.no_grad():
                # for each layer of the server model
                for layer_idx, model_server_param in enumerate(list(model_server.parameters())):
                    # make a placeholder variable consisting of zeros but with the correct shape
                    new_param_server = torch.zeros_like(model_server_param).type(torch.FloatTensor).cuda()

                    # for each client model
                    for client_model_param in client_model_params:
                        # acces the weights in the correct layer given by layer_idx. and copy a proportion of the weight
                        # into the place holder variable
                        new_param_server += (1/num_client_subset) * client_model_param[layer_idx].data

                    # the placeholder variable becomes the new server_model param
                    model_server_param.copy_(new_param_server)

            # reset the saved model params
            print("avg_client_subset_loss", client_subset_loss/num_client_subset)
            client_subset_loss = 0
            client_model_params = []













