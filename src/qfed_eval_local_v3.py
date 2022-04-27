from global_model_eval_non_tensor import global_model_eval_non_tensor
from global_model_eval import global_model_eval
import torch
from model import Net, mlr
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

# def global_model_eval(state_dict ="saved_models/Qfed_manual_state_dict.pt",
#                       data_folder = "dataset/test_stored_as_tensor",
#                       parameters = None,
#                       num_test_clients = None,  # this is the indexing of the list so None means all
#                       get_loss = False,
#                       model=Net):

os.chdir("..")
data_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\test_stored_as_tensors"
txt_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist\data\img_lab_by_user\usernames_train.txt"
wandb.init()
wandb.config.update({"dataset_path":r'C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist'}, allow_val_change=True)

num_test_clients = 20000
name1="mlr_10_0.1"
state_dict_path = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived\{}_state_dict.pt".format(name1)
users_used_for_training = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived\{}_users.json".format(name1)

acc1, loss1, _ = global_model_eval_non_tensor(state_dict=state_dict_path,
                                            data_folder=txt_folder,
                                            num_test_clients=num_test_clients,
                                            get_loss=True,
                                            model=mlr,
                                            users_used_for_training=users_used_for_training)

print("qfed")
print("local acc mean:", np.mean(np.array(acc1)))
print("local acc std:", np.std(np.array(acc1)))

print("local loss mean:", np.mean(np.array(loss1)))
print("local loss std:", np.std(np.array(loss1)))


name2="mlr_10_0.0"
state_dict_path = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived\{}_state_dict.pt".format(name2)
users_used_for_training = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\archived\{}_users.json".format(name2)


acc2, loss2, _ = global_model_eval_non_tensor(state_dict=state_dict_path,
                                            data_folder=txt_folder,
                                            num_test_clients=num_test_clients,
                                            get_loss=True,
                                            model=mlr,
                                            users_used_for_training=users_used_for_training)

print("fedavg")
print("local acc mean:", np.mean(np.array(acc2)))
print("local acc std:", np.std(np.array(acc2)))

print("local loss mean:", np.mean(np.array(loss2)))
print("local loss std:", np.std(np.array(loss2)))





plt.hist(acc1, bins=50, alpha=0.3, label=name1)
plt.hist(acc2, bins=50, alpha=0.3, label=name2)
plt.title("test acc for mlr all classes on last {} clients".format(num_test_clients))
plt.legend()
plt.show()

plt.hist(loss1, bins=50, alpha=0.3, label=name1)
plt.hist(loss2, bins=50, alpha=0.3, label=name2)
plt.title("test loss for mlr all classes on last {} clients".format(num_test_clients))
plt.legend()
plt.show()



#
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