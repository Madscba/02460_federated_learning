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
hist = True
model = mlr
alpha = 0.35
bins = 50

#name1 = "FedAvg_drop_stragglersfalse_b8_mlr_10c"
name2 = "FedX_sigma=0.001_q=0.001"
#name3 = "FedX_mlr_b8_lr0.001_sigma-1_S-1_mu1.0_q0.001_10c"
name3 = "FedX_sigma=0.0001_q=0.001"

#labels = ["Fed-Avg", "FedX(q=0.01)", "FedX(q=0.001)"]
labels = ["FedX(sigma=0.001 q=0.001)", "FedX-DP(sigma=0.0001 q=0.001)"]
#labels = ["Fed-Avg", "FedX(q=0.001)"]#, "FedX(q=0.001)"]

#names = [name1, name2, name3]
names = [name2, name3]
#colors = ["indianred", "darkseagreen", "cornflowerblue"]
colors = ["cornflowerblue", "darkseagreen"]
#colors = ["indianred", "cornflowerblue"]

figure, axis =  plt.subplots(1,2, figsize=(16,8))
#figure1, axis1 = plt.subplots(1,2, figsize=(16,8))
#figure2, axis2 = plt.subplots(1,2, figsize=(12,4))
ax1, ax2 = axis
#ax3, ax4 = axis1
#ax5, ax6 = axis2
for color, name, label in zip(colors, names, labels):
    state_dict_path = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\{}_state_dict.pt".format(name)
    acc_and_loss_path = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\{}_list.json".format(name)
    users_used_for_training = r"\\wsl$\Ubuntu-22.04\home\karl\desktop\saved_models\{}_users.json".format(name)

    try:
        if redo_predictions:
            raise FileNotFoundError

        with open(acc_and_loss_path) as file:
            acc_and_loss = json.load(file)
            print("loading from", name+".json")
            print("number of obs in loaded file:", len(acc_and_loss[0]))
            acc = acc_and_loss[0]
            loss = acc_and_loss[1]

    except FileNotFoundError:

        if not wandb_init:
            wandb.init()
            wandb.config.update({"dataset_path": r'C:\Users\Karlu\Desktop\advanced\02460_federated_learning\dataset\femnist'},allow_val_change=True)
            wandb_init = True

        print("predicting for.", name)
        acc, loss, _, confuss, pred_dist = global_model_eval_non_tensor(state_dict=state_dict_path,
                                                    data_folder=txt_folder,
                                                    num_test_clients=num_test_clients,
                                                    get_loss=True,
                                                    model=model,
                                                    train_data=False,
                                                    train_proportion=0.8, # should be 0.8 for train_data= False
                                                    num_classes = None,
                                                    users_used_for_training=None)

        confuss_diag = (confuss * (1 / (confuss.sum(axis=1).reshape(-1, 1) + 1))).diagonal()
        print("digits avg performance:", np.mean(confuss_diag[:10]))
        print("capital letters avg performance:", np.mean(confuss_diag[10:36]))
        print("small letters avg performance:", np.mean(confuss_diag[36:]))
        # if label == "Fed-Avg":
        #     print(confuss.sum(axis=1))
        #     pcm = axis1[0].imshow(confuss * (1/(confuss.sum(axis=1).reshape(-1,1)+1)), label=label)
        #     axis1[0].set_title("Fed-Avg")
        #     axis1[0].set_xlabel("True Label")
        #     axis1[0].set_ylabel("Prediction")
        #
        # else:
        #     print(confuss.sum(axis=1))
        #     pcm = axis1[1].imshow(confuss * (1/(confuss.sum(axis=1).reshape(-1,1)+1)), label=label)
        #     axis1[1].set_title("FedX")
        #     figure1.colorbar(pcm, ax=axis1)
        #
        # ax5.plot(pred_dist, color=color, label=label)
        # #ax5.set_title("Prediction distribution")
        # ax5.set_ylabel("Model confidence")
        # ax5.set_xlabel("Least to most likely class")
        # ax5.legend()

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
                 hist_kws={"alpha": alpha, "rwidth": 0.9},
                 ax=ax1, label=label + " - std: " + str(np.std(np.array(acc)))[:4])

    sns.distplot(loss, hist=hist, kde=True, bins=bins, color=color,
                 kde_kws={"alpha": 1, 'linewidth': 4, 'clip': (0.0, max(loss)*1.05)},
                 hist_kws={"alpha": alpha, "rwidth": 0.9},
                 ax=ax2, label=label + " - std: " + str(np.std(np.array(loss)))[:4])

    ax1.axvline(x=np.mean(np.array(acc)), color=color, linewidth=4, linestyle="dashed", alpha=1,
                label=label + " - mean: " + str(np.mean(np.array(acc)))[:4])
    ax2.axvline(x=np.mean(np.array(loss)), color=color, linewidth=4, linestyle="dashed", alpha=1,
                label=label + " - mean: " + str(np.mean(np.array(loss)))[:4])

ax1.set_title("Accuracy distribution")
ax2.set_title("Loss distribution")
ax2.set_ylabel("")
ax1.set_xlabel("Accuracy")
ax2.set_xlabel("Loss")

# from matplotlib.lines import Line2D
# custom_line = Line2D([0], [0], color="gray", lw=4, linestyle="dashed", alpha=alpha)
#
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# labels1.append("Mean")
# labels2.append("Mean")
#
# handles1.append(custom_line)
# handles2.append(custom_line)

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

# ax2.legend()
plt.tight_layout()
# plt.subplots_adjust(top=0.87)
# fig.suptitle("CNN client test performance distribution - unrestricted client classes", fontsize=20)
plt.savefig(os.path.basename(__file__)[:-3] + '.png', dpi=200)
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