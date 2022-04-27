from torch.utils.data import DataLoader
from client_dataset import FemnistDataset
import torchvision.transforms as transforms
import torch
from model import Net, mlr
from collections import OrderedDict
import os
from tqdm import tqdm
from main_utils import set_seed
import time
import json


def global_model_eval_non_tensor(state_dict =None,
                                 data_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist\data\img_lab_by_user\usernames_train.txt",
                                 num_test_clients = None,  # this is the indexing of the list so None means all
                                 get_loss = False,
                                 model=Net,
                                 users_used_for_training = None
                                 ):

    loss_func = torch.nn.CrossEntropyLoss()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model().to(DEVICE)
    #fc1.weight, fc1.bias
    # try:
    net.load_state_dict(torch.load(state_dict))
    #     print("loaded NN\n")
    #
    # except Exception:
    #     state_dict = torch.load(state_dict)
    #     state_dict["fc1.weight"] = state_dict["conv1.weight"]
    #     state_dict["fc1.bias"] = state_dict["conv1.bias"]
    #     state_dict.pop("conv1.weight")
    #     state_dict.pop("conv1.bias")
    #     print("loaded mlr\n")

    if users_used_for_training == None:
        with open(data_folder) as file: user_names_test = [line.strip() for line in file][:-1]
    else:
        with open(users_used_for_training) as file:
            user_names_test = json.load(file)

    print(len(user_names_test))
    user_names_test = user_names_test[-num_test_clients:]
    print(len(user_names_test))
    #print(user_names_test[-5:])

    transform = transforms.Compose([transforms.ToTensor()])
    acc, loss, num_obs_per_user = [], [], []
    with torch.no_grad():
        for user in tqdm(user_names_test):
            #k+=1
            dataset = FemnistDataset(user, transform, train=False, train_proportion=0.8)
            data_loader = DataLoader(dataset, batch_size=80000)
            for x, y in data_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                num_obs_per_user.append(x.shape[0])
                pred = net(x)
                acc.append(torch.mean((torch.argmax(pred, axis=1) == y).type(torch.float)).item() * 100)

                if get_loss:
                    loss.append(loss_func(pred, y).item())

            #if verbose: print("client", k, "time:", str(time.time() - t)[:5])
    return acc, loss, num_obs_per_user

# if __name__ == '__main__':
    # import os
    # import numpy as np
    # import time
    # os.chdir("..")
    # print(os.getcwd())
    #
    # state_dict = "saved_models/fedavg_state_dict.pt"
    # user_names_test_file = "dataset/femnist/data/img_lab_by_user/user_names_test.txt"
    # num_test_clients = 100 # i.e. the 10 first
    # get_loss = True
    #
    # t = time.time()
    # acc, loss, num_obs_per_user = global_model_eval(state_dict, user_names_test_file, num_test_clients, get_loss)
    # print("time:", time.time()-t)
    # print("mean_acc:", np.mean(np.asarray(acc)))
    # print(loss)
    # print(num_obs_per_user)





