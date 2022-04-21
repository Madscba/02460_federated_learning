from global_model_eval import global_model_eval
import os
import numpy as np
import time


if __name__ == "__main__":
    os.chdir("..")
    print(os.getcwd())

    state_dict = "saved_models/Fedavg_state_dict_16_15_48.pt"
    user_names_test_file = "dataset/femnist/data/img_lab_by_user/usernames_test.txt"
    num_test_clients = 100 # i.e. the 100 first
    get_loss = True

    t = time.time()
    acc, loss, num_obs_per_user = global_model_eval(state_dict, user_names_test_file, num_test_clients, get_loss)
    print("time:", time.time()-t)
    print("mean_acc:", np.mean(np.asarray(acc)))
    print(loss)
    print(num_obs_per_user)


