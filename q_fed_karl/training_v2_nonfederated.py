from cifar10_noniid import get_dataset_cifar10_extr_noniid
from network import cifar_net
import torch.optim as optim
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch


# data
num_clients = 50
nclass = 8
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
optimizer = optim.Adam(model_server.parameters(), lr=lr)
total_samples = 50000


# training hyper params:
server_epochs = 100
batch_size = 40 # should have "total_samples_client % batch_size = 0"

for server_epoch in range(server_epochs):
    n_eval_points = 1000
    random_perm = torch.randperm(10000)[:n_eval_points]
    # get test acc across all testdata :
    test_pred = model_server.forward(test_data[random_perm]).detach()
    print("\ntest acc:", (torch.sum(torch.argmax(test_pred, 1) == test_targets[random_perm])/n_eval_points).item())

    # get train acc over a random selection of the train data
    train_pred = model_server.forward(train_data[random_perm*5]).detach()
    print("train acc:", (torch.sum(torch.argmax(train_pred, 1) == train_targets[random_perm*5])/n_eval_points).item())

    random_perm = torch.randperm(total_samples)
    train_data = train_data[random_perm]
    train_targets = train_targets[random_perm]

    for batch_idx in range(total_samples // batch_size):
        optimizer.zero_grad()
        batch_x = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_y = train_targets[batch_idx*batch_size:(batch_idx+1)*batch_size]

        pred = model_server.forward(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()








