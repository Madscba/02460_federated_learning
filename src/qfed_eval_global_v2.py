from global_model_eval import global_model_eval

import torch
from model import Net
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np

# def global_model_eval(state_dict ="saved_models/Qfed_manual_state_dict.pt",
#                       data_folder = "dataset/test_stored_as_tensor",
#                       parameters = None,
#                       num_test_clients = None,  # this is the indexing of the list so None means all
#                       get_loss = False,
#                       model=Net):

os.chdir("..")
data_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\test_stored_as_tensors"
num_test_clients = None

state_dict_dir = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived"
state_dicts = sorted(os.listdir(state_dict_dir))
state_dict_paths = [os.path.join(state_dict_dir, state_dict) for state_dict in state_dicts if state_dict.endswith(".pt")]
acc_paths = [os.path.join(state_dict_dir, "acc_global", state_dict[:-3]) for state_dict in state_dicts]
loss_paths = [os.path.join(state_dict_dir, "loss_global", state_dict[:-3]) for state_dict in state_dicts]

make_predictions = True

if make_predictions:
    for state_dict_path, acc_path, loss_path in zip(state_dict_paths, acc_paths, loss_paths):
        acc, loss, _ = global_model_eval(state_dict=state_dict_path,
                                         data_folder=data_folder,
                                         num_test_clients=num_test_clients,
                                         get_loss=True,
                                         verbose=False)


        np.save(acc_path, np.array(acc))
        np.save(loss_path, np.array(loss))
        print(state_dict_path)
        print("mean:", np.mean(np.array(acc)))
        print("std:", np.std(np.array(acc)))
        print("\n")



# # acc1 = np.load('data.npy')
# # acc2 = np.load('data.npy')
#
# acc1 = np.load(data_folder + predict1_name)
# acc2 = np.load(data_folder + predict2_name)
# loss1 = np.load(data_folder + loss1_name)
# loss2 = np.load(data_folder + loss2_name)
#
# print("loss q{} mean and std".format(q1), np.mean(loss1), np.std(loss1))
# print("loss q{} mean and std".format(q2), np.mean(loss2), np.std(loss2))
#
# print("acc q{} mean and std".format(q1), np.mean(acc1), np.std(acc1))
# print("acc q{} mean and std".format(q2), np.mean(acc2), np.std(acc2))
#
#
# plt.hist(acc1, bins=100, color="red", label="q={}".format(q1), alpha=0.5)
# plt.hist(acc2, bins=100, color="green", label="q={}".format(q2), alpha=0.5)
# plt.title("prediction acc")
# plt.legend()
# plt.show()
#
# plt.hist(loss1, bins=100, color="red", label="q={}".format(q1), alpha=0.5)
# plt.hist(loss2, bins=100, color="green", label="q={}".format(q2), alpha=0.5)
# plt.title("losses")
# plt.legend()
# plt.show()