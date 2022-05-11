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
import numpy as np


def global_model_eval_non_tensor(state_dict =None,
                                 data_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist\data\img_lab_by_user\usernames_train.txt",
                                 num_test_clients = None,  # this is the indexing of the list so None means all
                                 get_loss = False,
                                 model=Net,
                                 users_used_for_training = None,
                                 train_data=False,
                                 train_proportion=0.8,
                                 num_classes=None,
                                 DEVICE = None):

    sorted_dist = np.zeros(62)
    confuss = np.zeros((62,62))
    loss_func = torch.nn.CrossEntropyLoss()
    if not DEVICE:
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
    user_names_test = set(user_names_test[-num_test_clients:])
    print(len(user_names_test))
    #print(user_names_test[-5:])

    transform = transforms.Compose([transforms.ToTensor()])
    acc, loss, num_obs_per_user = [], [], []
    with torch.no_grad():
        for user in tqdm(user_names_test):
            #k+=1
            dataset = FemnistDataset(user, transform,
                                     train=train_data, train_proportion=train_proportion, num_classes=num_classes)
            data_loader = DataLoader(dataset, batch_size=80000)
            for x, y in data_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                #softmax = torch.nn.Softmax(dim=1)
                num_obs_per_user.append(x.shape[0])
                pred = net(x)
                pred_arg_max = torch.argmax(pred, axis=1)
                pred_numpy = pred.cpu().numpy()
                pred_numpy.sort(axis=1) # this one is inplace
                sorted_dist += pred_numpy.sum(axis=0)

                for pred_, y_ in zip(pred_arg_max.cpu().numpy(), y.cpu().numpy()): confuss[pred_, y_] += 1

                acc.append((torch.mean((pred_arg_max == y).type(torch.float))).item() * 100)

                if get_loss:
                    loss.append(loss_func( pred, y).item())

            #if verbose: print("client", k, "time:", str(time.time() - t)[:5])
    return acc, loss, num_obs_per_user, confuss, sorted_dist

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





