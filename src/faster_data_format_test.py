from torch.utils.data import DataLoader
from client_dataset import FemnistDataset
import torchvision.transforms as transforms
import torch
from model import Net
from tqdm import tqdm

def global_model_eval(state_dict ="Qfed_manual_state_dict.pt",
                      state_dict_folder = "saved_models",
                      data_folder = "data_stored_as_tensors",
                      num_test_clients = None,  # this is the indexing of the list so None means all
                      get_loss = False):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    net = Net().to(DEVICE)
    cwd = os.getcwd()
    state_dict_path = os.path.join(cwd, state_dict_folder, state_dict)
    net.load_state_dict(torch.load(state_dict_path))
    if get_loss: loss_func = torch.nn.CrossEntropyLoss()

    client_names = os.listdir(os.path.join(cwd, data_folder))
    x_data = sorted([os.path.join(cwd, data_folder, client)
                     for client in client_names if client.endswith("x.pt")])[:num_test_clients]

    y_data = sorted([os.path.join(cwd, data_folder, client)
                     for client in client_names if client.endswith("y.pt")])[:num_test_clients]

    acc, loss, num_obs_per_user = [], [], []
    with torch.no_grad():
        for x, y in tqdm(zip(x_data, y_data)):
            x = torch.load(x).to(DEVICE)
            y = torch.load(y).to(DEVICE)
            num_obs_per_user.append(x.shape[0])

            #batch_size = 8
            batch_size = num_obs_per_user[-1] # use all the data
            # depending on the batch size this is a loop of one iteration
            for i in range(num_obs_per_user[-1] // batch_size):
                x_ = x[i*batch_size:  i*batch_size+batch_size]
                y_ = y[i*batch_size:  i*batch_size+batch_size]

                pred = net(x_)
                acc.append(torch.mean((torch.argmax(pred, axis = 1) == y_).type(torch.float)).item()*100)

                if get_loss:
                    loss.append(loss_func(pred, y_).item())

    return acc, loss, num_obs_per_user

if __name__ == '__main__':
    import os
    import numpy as np
    import time
    os.chdir("..")
    print(os.getcwd())

    state_dict = "Fedavg_state_dict.pt"
    state_dict_folder = "saved_models"
    user_names_test_file = "data_stored_as_tensors"
    num_test_clients = 100 # i.e. the 10 first
    get_loss = True

    t = time.time()
    acc, loss, num_obs_per_user = global_model_eval(state_dict = "Fedavg_state_dict.pt",
                                                    state_dict_folder = "saved_models",
                                                    data_folder = "data_stored_as_tensors",
                                                    num_test_clients = 100, # i.e. the 10 first
                                                    get_loss = True)

    print("time:", time.time()-t)
    print("mean_acc:", np.mean(np.asarray(acc)))
    print(loss)
    print(num_obs_per_user)





