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

os.chdir("../../..")
print(os.getcwd())
data_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\test_stored_as_tensors"
num_test_clients = None
q1 = "0.0"
q2 = "0.002"

predict1_name = "/predictions_q_{}.npy".format(q1)
predict2_name = "/predictions_q_{}.npy".format(q2)

loss1_name = "/loss_q_{}.npy".format(q1)
loss2_name = "/loss_q_{}.npy".format(q2)

state_dict1 = "saved_models/Qfed_manual_{}_state_dict.pt".format(q1)
state_dict2 = "saved_models/Qfed_manual_{}_state_dict.pt".format(q2)
make_predictions = False

if make_predictions:
    acc1, loss1, num_obs_per_user = global_model_eval(state_dict=state_dict1,
                                                    data_folder=data_folder,
                                                    num_test_clients=num_test_clients,
                                                    get_loss=True,
                                                    verbose=True)


    acc2, loss2, num_obs_per_user = global_model_eval(state_dict=state_dict2,
                                                    data_folder=data_folder,
                                                    num_test_clients=num_test_clients,
                                                    get_loss=True,
                                                    verbose=True)

    #predictions
    np.save(data_folder + predict1_name, np.array(acc1))
    np.save(data_folder + predict2_name, np.array(acc2))

    #losses
    np.save(data_folder + loss1_name, np.array(loss1))
    np.save(data_folder + loss2_name, np.array(loss2))


# acc1 = np.load('data.npy')
# acc2 = np.load('data.npy')

acc1 = np.load(data_folder + predict1_name)
acc2 = np.load(data_folder + predict2_name)
loss1 = np.load(data_folder + loss1_name)
loss2 = np.load(data_folder + loss2_name)

from tqdm import tqdm

def bootstrap(val1, val2, fn = np.std, name="loss", quantilies=[0.025, 0.975], n = 10000):
    stds1 = []
    stds2 = []
    stds_diff = []
    N = val1.shape[0]
    loss_diff = val2-val1
    for _ in tqdm(range(n)):
        indx = np.round((np.random.uniform(size=N)*(N-1))).astype(int)
        stds1.append(fn(val1[indx]))
        stds2.append(fn(val2[indx]))
        stds_diff.append(fn(loss_diff[indx]))
    
    print(name, fn(val1), np.quantile(np.array(stds1), quantilies))
    print(name, fn(val2), np.quantile(np.array(stds2), quantilies))
    print(name, fn(loss_diff), np.quantile(np.array(stds_diff), quantilies))
   
bootstrap(acc1, acc2, np.mean, "acc")
bootstrap(loss1, loss2, np.mean, "loss")

 




print("loss q{} mean and std".format(q1), np.mean(loss1), np.std(loss1))
print("loss q{} mean and std".format(q2), np.mean(loss2), np.std(loss2))

print("acc q{} mean and std".format(q1), np.mean(acc1), np.std(acc1))
print("acc q{} mean and std".format(q2), np.mean(acc2), np.std(acc2))


plt.hist(acc1, bins=100, color="red", label="q={}".format(q1), alpha=0.5)
plt.hist(acc2, bins=100, color="green", label="q={}".format(q2), alpha=0.5)
plt.title("prediction acc")
plt.legend()
plt.show()

plt.hist(loss1, bins=100, color="red", label="q={}".format(q1), alpha=0.5)
plt.hist(loss2, bins=100, color="green", label="q={}".format(q2), alpha=0.5)
plt.title("losses")
plt.legend()
plt.show()