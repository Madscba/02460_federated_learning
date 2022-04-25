from torch.utils.data import DataLoader
from client_dataset import FemnistDataset
import torchvision.transforms as transforms
import torch
from model import Net
from collections import OrderedDict
import os


def global_model_eval(state_dict ="saved_models/Qfed_manual_state_dict.pt",
                      data_folder = "dataset/test_stored_as_tensor",
                      parameters = None,
                      num_test_clients = None,  # this is the indexing of the list so None means all
                      get_loss = False, 
                      model=Net,
                      verbose=False):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model().to(DEVICE)
    if parameters:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    else:
        net.load_state_dict(torch.load(state_dict))

    if get_loss: loss_func = torch.nn.CrossEntropyLoss()

    client_names = os.listdir(data_folder)
    x_data = sorted([os.path.join(data_folder, client)
                     for client in client_names if client.endswith("x.pt")])[:num_test_clients]
    y_data = sorted([os.path.join(data_folder, client)
                     for client in client_names if client.endswith("y.pt")])[:num_test_clients]

    import time
    t = time.time()
    acc, loss, num_obs_per_user = [], [], []
    with torch.no_grad():
        k = 0
        for x, y in zip(x_data, y_data):
            k+=1
            x = torch.load(x, map_location=torch.device('DEVICE') ).to(DEVICE)
            y = torch.load(y, map_location=torch.device('DEVICE') ).to(DEVICE)
            num_obs_per_user.append(x.shape[0])

            # batch_size = 8
            # use all the data instead of an actual batch size
            batch_size = num_obs_per_user[-1]
            for i in range(num_obs_per_user[-1] // batch_size):
                x_ = x[i * batch_size:i * batch_size + batch_size]
                y_ = y[i * batch_size:i * batch_size + batch_size]

                pred = net(x_)
                acc.append(torch.mean((torch.argmax(pred, axis=1) == y_).type(torch.float)).item() * 100)

                if get_loss:
                    loss.append(loss_func(pred, y_).item())

            if verbose: print("client", k, "time:", str(time.time() - t)[:5])
    return acc, loss, num_obs_per_user

if __name__ == '__main__':
    import os
    import numpy as np
    import time
    os.chdir("..")
    print(os.getcwd())

    state_dict = "saved_models/fedavg_state_dict.pt"
    user_names_test_file = "dataset/femnist/data/img_lab_by_user/user_names_test.txt"
    num_test_clients = 100 # i.e. the 10 first
    get_loss = True

    t = time.time()
    acc, loss, num_obs_per_user = global_model_eval(state_dict, user_names_test_file, num_test_clients, get_loss)
    print("time:", time.time()-t)
    print("mean_acc:", np.mean(np.asarray(acc)))
    print(loss)
    print(num_obs_per_user)





