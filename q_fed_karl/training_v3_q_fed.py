from cifar10_noniid import get_dataset_cifar10_extr_noniid
from network import cifar_net
import torch.optim as optim
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt


# data
num_clients = 100
nclass = 5
nsamples = int(50000//(nclass*num_clients))
total_samples_client = int(nclass*nsamples)

rate_unbalance = 1.0
train_dataset, test_dataset, client_groups_train, client_groups_test = \
    get_dataset_cifar10_extr_noniid(num_clients, nclass, nsamples, rate_unbalance)


# train data
train_data = torch.tensor(train_dataset.data/255).type(torch.FloatTensor).cuda()
train_data = torch.reshape(train_data, (50000, 3, 32, 32))
train_targets = torch.tensor(train_dataset.targets).type(torch.LongTensor).cuda()

# test data
test_data = torch.tensor(test_dataset.data/255).type(torch.FloatTensor).cuda()
test_data = torch.reshape(test_data, (10000, 3, 32, 32))
test_targets = torch.tensor(test_dataset.targets).type(torch.LongTensor).cuda()

# client idx
for key, client_data in list(client_groups_train.items()):
    client_groups_train[key] = torch.tensor(client_groups_train[key]).type(torch.LongTensor).cuda()

for key, client_data in list(client_groups_test.items()):
    client_groups_test[key] = torch.tensor(client_groups_test[key]).type(torch.LongTensor).cuda()


# network and optimizer
model_server = cifar_net()
model_server.cuda()
criterion = nn.CrossEntropyLoss()
lr = 0.001
L = 1 / lr


# training hyper params:
server_epochs = 50
batch_size = 40 # should have "total_samples_client % batch_size = 0"
num_client_subset = 10 # number of clients to learn from before updating the main model
client_epochs = 2 # number of epoch to run per client


def norm(list_of_grads):
    n = torch.tensor(0, dtype = torch.float).cuda()
    for grad in list_of_grads:
        n += torch.norm(grad)

    return n
# #profiling
# import cProfile
# import pstats
#
# with cProfile.Profile() as pr:
hs = torch.tensor(0, dtype = torch.float).cuda()
deltas = [torch.zeros_like(param).type(torch.FloatTensor).cuda() for param in list(model_server.parameters())]
eps = 1e-10 # for numerical stability
q = 2


for server_epoch in range(server_epochs):
    training_order = np.random.permutation(num_clients)

    # get model performance on test and train set
    # we dont care for these gradients
    with torch.no_grad():
        n_eval_points = 1000
        random_perm = torch.randperm(10000)[:n_eval_points] # 10000 = number of test obs
        # get test acc across all testdata :
        test_pred = model_server.forward(test_data[random_perm])
        print("\nepoch {}:".format(server_epoch))
        print("test acc:", (torch.sum(torch.argmax(test_pred, 1) == test_targets[random_perm])/n_eval_points).item())

        # get train acc over a random selection of the train data
        train_pred = model_server.forward(train_data[random_perm*5])
        print("train acc:", (torch.sum(torch.argmax(train_pred, 1) == train_targets[random_perm*5])/n_eval_points).item())

    for i, client in enumerate(training_order):
        #print(i)
        model_client = deepcopy(model_server)
        optimizer = optim.Adam(model_client.parameters(), lr=lr)

        # get client specific data
        client_data_x = train_data[client_groups_train[client]]
        client_target = train_targets[client_groups_train[client]]

        # calculate the server models loss on the clients training data:
        with torch.no_grad():
            loss_s = (criterion(model_server.forward(client_data_x), client_target))

        # the actual client training loop
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
                loss_c = criterion(pred, batch_y) # loss_c = client loss
                loss_c.backward()
                optimizer.step()

        # performing the actual algorithm as given in the paper
        with torch.no_grad(): # for good measure : )
            # save the client model params after the epoch
            client_model_params = deepcopy(list(model_client.parameters()))
            server_model_params = deepcopy(list(model_server.parameters()))

            # i am not entirely sure why the gradients are calcualted this way. maybe i could try with
            # model.param.grads, or whatever its called.
            grads = [(weights_before - weights_after) * L for weights_before, weights_after in
                     zip(server_model_params, client_model_params)]


            deltas = [delta + torch.float_power(loss_s + eps, q) * grad for delta, grad in zip(deltas,grads)]

            hs += q * torch.float_power(loss_s + eps, q-1) * norm(grads) + L * torch.float_power(loss_s + eps, q)

        # update server model every num_client_subset according to the algorithm
        if (1+i) % num_client_subset == 0:
            # make sure to not track the gradient
            with torch.no_grad():
                for delta, model_server_param in zip(deltas, list(model_server.parameters())):
                    # not sure if cloning is needed or not. lets be save
                    model_server_param.copy_(model_server_param.data - (delta / hs))

                # reset params
                hs = torch.tensor(0, dtype = torch.float).cuda()
                deltas = [torch.zeros_like(param).type(torch.FloatTensor).cuda() for param in list(model_server.parameters())]

accs = []
n_samples = client_groups_train[0].shape[0]
for client_data in list(client_groups_train.values()):
    with torch.no_grad():
        # torch.sum(torch.argmax(test_pred, 1) == test_targets[random_perm]
        pred = model_server.forward(train_data[client_data])
        acc = torch.sum(torch.argmax(pred, 1) == train_targets[client_data]) / n_samples
        accs.append(acc.item())

plt.hist(accs)
plt.savefig("acc_hist_with_q_{}.png".format(q))
plt.show()



    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10)













