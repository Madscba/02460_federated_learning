from torch.utils.data import DataLoader
from client_dataset import FemnistDataset
import torchvision.transforms as transforms
import torch
from model import Net
from tqdm import tqdm


def global_model_eval(state_dict="saved_models/Qfed_manual_state_dict.pt",
                      user_names_test_file="dataset/femnist/data/img_lab_by_user/user_names_train.txt",
                      num_test_clients=None,  # this is the indexing of the list so None means all
                      get_loss=False):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(DEVICE)
    net.load_state_dict(torch.load(state_dict))
    loss_func = torch.nn.CrossEntropyLoss()

    with open(user_names_test_file) as file:
        user_names_test = [line.strip() for line in file]
    print(len(user_names_test))
    user_names_test = user_names_test[:num_test_clients]
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    acc, loss, num_obs_per_user = [], [], []
    with torch.no_grad():
        for user in tqdm(user_names_test):
            dataset = FemnistDataset(user, transform, train=True, train_proportion=1)
            # set arbitrary big batch size such that we only get one batch
            # with all the data
            data_loader = DataLoader(dataset, batch_size=8000)

            for x, y in data_loader:
                torch.save(x.type(torch.float16), "dataset/train_stored_as_tensors_test/" + user + "_x.pt")
                torch.save(y.type(torch.float16), "dataset/train_stored_as_tensors_test/" + user + "_y.pt")

                # x, y = x.to(DEVICE), y.to(DEVICE)
                # num_obs_per_user.append(x.shape[0])
                #     pred = net(x)
                #     acc.append(torch.mean((torch.argmax(pred, axis = 1) == y).type(torch.float)).item()*100)
                #
                #     if get_loss:
                #         loss.append(loss_func(pred, y).item())

    return acc, loss, num_obs_per_user


if __name__ == '__main__':
    import os
    import numpy as np
    import time

    os.chdir("..")
    print(os.getcwd())

    import wandb

    dataset_path = None  # '/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'
    wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
    wandb.init(project="02460_federated_learning", entity="02460-federated-learning")
    wandb.config.update({'dataset_path': dataset_path})

    state_dict = "saved_models/fedavg_state_dict.pt"
    user_names_test_file = "dataset/femnist/data/img_lab_by_user/user_names_train.txt"
    num_test_clients = 2  # i.e. the 10 first
    get_loss = True

    t = time.time()
    acc, loss, num_obs_per_user = global_model_eval(state_dict, user_names_test_file, num_test_clients, get_loss)
    print("time:", time.time() - t)
    print("mean_acc:", np.mean(np.asarray(acc)))
    print(loss)
    print(num_obs_per_user)





