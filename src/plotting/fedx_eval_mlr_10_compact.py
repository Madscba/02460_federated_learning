import os
os.chdir("../..")
from global_model_eval_non_tensor import global_model_eval_non_tensor
from model import Net, mlr
import matplotlib.pyplot as plt
import numpy as np
import wandb
import json
import seaborn as sns
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
font = {
        'weight' : 'bold',
        'size'   : 12
       }

import matplotlib
matplotlib.rc('font', **font)

data_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\test_stored_as_tensors"
txt_folder = r"C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist\data\img_lab_by_user\usernames_train.txt"
wandb_init = False

num_test_clients = 10000
redo_predictions = False
model = mlr
alpha = 0.35
hist = True
bins = 50

name1 = "NO_NAME"
name2 = "FedX_true_S1.0_q0.01"
#name3 = "qfed_strag_1000_rounds_0.01"
labels = [ "Fed-Avg", "FedX"]
#labels = [ "Fed-Avg", "FedX", "Qfed-Ours"]

#names = [name1, name2, name3]
names = [name1, name2]
#colors = ["indianred", "darkseagreen", "cornflowerblue"]
colors = ["indianred", "cornflowerblue"]

fig, ax1 = plt.subplots(1,1, figsize=(7,4))
#ax1, ax2 = ax
for color, name, label in zip(colors, names, labels):
    state_dict_path = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\{}_state_dict.pt".format(name)
    acc_and_loss_path = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\{}_list.json".format(name)
    #users_used_for_training = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\qfed_strag_1000_rounds_0.0_users.json"

    try:
        if redo_predictions:
            raise FileNotFoundError

        with open(acc_and_loss_path) as file:
            acc_and_loss = json.load(file)
            print("loading from", name1+".json")
            print("number of obs in loaded file:", len(acc_and_loss[0]))
            acc = acc_and_loss[0]
            loss = acc_and_loss[1]

    except FileNotFoundError:

        if not wandb_init:
            wandb.init()
            wandb.config.update({"dataset_path": r'C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist'},allow_val_change=True)
            wandb_init = True

        print("predicting for.", name)
        acc, loss, _ = global_model_eval_non_tensor(state_dict=state_dict_path,
                                                    data_folder=txt_folder,
                                                    num_test_clients=num_test_clients,
                                                    get_loss=True,
                                                    model=model,
                                                    users_used_for_training=None)

        acc_and_loss = [acc, loss]
        with open(acc_and_loss_path, "w") as file:
            print("saving to list")
            json.dump([acc, loss], file)

    print("local acc mean:", np.mean(np.array(acc)))
    print("local acc std:", np.std(np.array(acc)))

    print("local loss mean:", np.mean(np.array(loss)))
    print("local loss std:", np.std(np.array(loss)), "\n")

    sns.distplot(acc, hist=hist, kde=True, bins=bins, color=color,
                 kde_kws={"alpha": 1, 'linewidth': 4, 'clip': (0.0, 100)},
                 hist_kws={"alpha": alpha, "rwidth":0.9},
                 ax=ax1, label=label+" - std: " + str(np.std(np.array(acc)))[:4])

    # sns.distplot(loss, hist=hist, kde=True, bins=bins, color=color,
    #              kde_kws={"alpha": 1, 'linewidth': 4},
    #              hist_kws={"alpha": alpha, "rwidth":0.9},
    #              ax=ax2, label=label+" - std: " + str(np.std(np.array(loss)))[:4])

    ax1.axvline(x=np.mean(np.array(acc)), color=color, linewidth=4, linestyle="dashed", alpha=1,
                label=label+" - mean: " + str(np.mean(np.array(acc)))[:4])
    # ax2.axvline(x=np.mean(np.array(loss)), color=color, linewidth=4, linestyle="dashed", alpha=1,
    #             label=label+" - mean: " + str(np.mean(np.array(loss)))[:4])

#ax1.set_title("Accuracy distribution")
#ax2.set_title("Loss distribution")
#ax2.set_ylabel("")
ax1.set_ylabel("")
ax1.set_xlabel("Accuracy distribution")
ax1.xaxis.set_label_position('top')
##ax2.xaxis.set_label_position('top')
#ax2.set_xlabel("Loss distribution")
#ax2.set_xlim(-0.5, 6.2)


ax1.legend(loc="upper right")
#ax2.legend(loc="upper right")

#ax2.legend()

#ax2.legend()
plt.tight_layout()
#plt.subplots_adjust(top=0.87)
#fig.suptitle("", fontsize=18)
plt.savefig(os.path.basename(__file__)[:-3]+'true_single.png', dpi=200)
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
# print("loss q{} mean and var".format(q1), np.mean(loss1), np.var(loss1))
# print("loss q{} mean and var".format(q2), np.mean(loss2), np.var(loss2))
#
# print("acc q{} mean and var".format(q1), np.mean(acc1), np.var(acc1))
# print("acc q{} mean and var".format(q2), np.mean(acc2), np.var(acc2))
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