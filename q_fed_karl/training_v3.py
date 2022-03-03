from cifar10_noniid import get_dataset_cifar10_extr_noniid
from network import cifar_net
import torch.optim as optim
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch


# data
num_clients = 50
nclass = 10
nsamples = int(50000//(nclass*num_clients))
total_samples_client = int(nclass*nsamples)

rate_unbalance = 1.0
train_dataset, test_dataset, client_groups_train, client_groups_test = \
    get_dataset_cifar10_extr_noniid(num_clients, nclass, nsamples, rate_unbalance)



train_data = torch.tensor(train_dataset.data/255).type(torch.FloatTensor).cuda()
train_data = torch.reshape(train_data, (50000, 3, 32, 32))
train_targets = torch.tensor(train_dataset.targets).type(torch.LongTensor).cuda()

for key, val in list(client_groups_train.items()):
    client_groups_train[key] = torch.tensor(client_groups_train[key]).type(torch.LongTensor).cuda()

# test data
#test_data = torch.tensor(test_dataset.data/255).type(torch.FloatTensor).cuda()
#test_targets = torch.tensor(test_dataset.targets).type(torch.FloatTensor).cuda()
test_data = torch.tensor(test_dataset.data/255).type(torch.FloatTensor).cuda()
test_data = torch.reshape(test_data, (10000, 3, 32, 32))
test_targets = torch.tensor(test_dataset.targets).type(torch.LongTensor).cuda()



# network and optimizer
model_server = cifar_net()
model_server.cuda()
criterion = nn.CrossEntropyLoss()
lr = 0.001


# training hyper params:
server_epochs = 100
batch_size = 40 # should have "total_samples_client % batch_size = 0"
num_client_subset = 10 # number of clients to learn from before updating the main model
client_epochs = 1 # number of epoch to run per client


# #profiling
# import cProfile
# import pstats
#
# with cProfile.Profile() as pr:
for server_epoch in range(server_epochs):
    training_order = np.random.permutation(num_clients)

    client_model_params = []
    client_subset_loss = 0

    n_eval_points = 1000
    random_perm = torch.randperm(10000)[:n_eval_points]
    # get test acc across all testdata :
    test_pred = model_server.forward(test_data[random_perm]).detach()
    print("\ntest acc:", (torch.sum(torch.argmax(test_pred, 1) == test_targets[random_perm])/n_eval_points).item())

    # get train acc over a random selection of the train data
    train_pred = model_server.forward(train_data[random_perm*5]).detach()
    print("train acc:", (torch.sum(torch.argmax(train_pred, 1) == train_targets[random_perm*5])/n_eval_points).item())

    for client_idx, client in enumerate(training_order):
        #print(client_idx)
        model_client = deepcopy(model_server)
        optimizer = optim.Adam(model_client.parameters(), lr=lr)

        # get client specific data
        client_data_x = train_data[client_groups_train[client]]
        client_target = train_targets[client_groups_train[client]]

        for _ in range(client_epochs):
            # shuffle the clients data each clientepoch:
            random_perm = torch.randperm(total_samples_client)

            # perform updates on each batch
            for batch_idx in range(total_samples_client // batch_size):
                optimizer.zero_grad()
                # extract one batch from all of the clients training data
                batch_x = client_data_x[random_perm[batch_idx*batch_size:(batch_idx+1)*batch_size]]
                batch_y = client_target[random_perm[batch_idx*batch_size:(batch_idx+1)*batch_size]]

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
            #print("avg_client_subset_loss", client_subset_loss/num_client_subset)
            client_subset_loss = 0
            client_model_params = []

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10)













