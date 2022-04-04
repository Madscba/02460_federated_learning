import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
sys.path.insert(0,"/work3/s173934/AdvML/02460_federated_learning/src/")
#print("Path: ",os.path.join(os.getcwd(),"src/data_exploration"))
from train_test_utils import load_data



class FEMNISTDatasetExplorer():
    def __init__(self,subset=False,subset_name="Train",username_filepath = "/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user"):
        self.username_filepath = username_filepath
        self.usernames_all = self._read_usernames_from_file(self.get_usernames_filepath())
        self.usernames_train = self._read_usernames_from_file(self.get_usernames_filepath(subset=True,subset_name="Train"))
        self.usernames_test = self._read_usernames_from_file(self.get_usernames_filepath(subset=True,subset_name="Test"))
        self.subset = subset
        self.subset_name = subset_name
        



    def _read_usernames_from_file(self,path):
        user_names_txt = open(path, 'r')
        user_names = user_names_txt.readlines()
        return [user.replace("\n","") for user in user_names]
        

    def get_usernames_filepath(self,subset=False,subset_name="Train"):
        if not subset:
            return os.path.join(self.username_filepath,"usernames_shuff_all.txt")
        else:
            if subset_name == "Train":
                return os.path.join(self.username_filepath,"usernames_train.txt")
            else:
                return os.path.join(self.username_filepath,"usernames_test.txt")
        
    def get_dataset_stats(self):
        photos_in_categories_total_train = []
        photos_in_categories_total_test = []

        if self.subset:
            if self.subset_name =="Train":
                users = self.usernames_train
            else:
                users = self.usernames_test
        else:
            users = self.usernames_all

        for user in tqdm(users):
            if user != "":
                photos_in_categories_train = np.zeros(62)
                photos_in_categories_test = np.zeros(62)

                trainloader, testloader, num_examples = load_data(user)
                #Run through train data
                for _, labels in trainloader:
                    for label in labels:
                        photos_in_categories_train[label-1] += 1
                photos_in_categories_total_train.append(photos_in_categories_train)
                #Run through test data
                for _, labels in testloader:
                    for label in labels:
                        photos_in_categories_test[label-1] += 1
                photos_in_categories_total_test.append(photos_in_categories_test)
        return photos_in_categories_total_train, photos_in_categories_total_test

if __name__ == "__main__":
    config=os.path.join("/work3/s173934/AdvML/02460_federated_learning/",'src','config',"config.yaml")
    wandb.login(key='47304b319fc295d13e84bba0d4d020fc41bd0629')
    wandb.init(project="02460_federated_learning", entity="02460-federated-learning", group="dummy", config=config, mode="disabled",job_type='client')

    #Get full dataset statistics
    # dataexplorer = FEMNISTDatasetExplorer()
    # photos_in_categories_all_train, photos_in_categories_all_test = dataexplorer.get_dataset_stats()
    # with open("dataset_stats_all_users_train_data.picl","wb") as f:
    #     pickle.dump(photos_in_categories_all_train, f)

    # with open("dataset_stats_all__users_test_data.picl","wb") as f:
    #     pickle.dump(photos_in_categories_all_test, f)

    # print("Succesful retrieving full data")
    #Get test dataset statistics
    # dataexplorer = FEMNISTDatasetExplorer(subset=True,subset_name="Test")
    # photos_in_categories_test_train, photos_in_categories_test_test = dataexplorer.get_dataset_stats()
    # with open("dataset_stats_test_users_train_data.picl","wb") as f:
    #     pickle.dump(photos_in_categories_test_train, f)

    # with open("dataset_stats_test_users_test_data.picl","wb") as f:
    #     pickle.dump(photos_in_categories_test_test, f)
    
    # print("Succesful retrieving test data")

    # #Get train dataset statistics
    # dataexplorer = FEMNISTDatasetExplorer(subset=True,subset_name="Train")
    # photos_in_categories_train_train, photos_in_categories_train_test = dataexplorer.get_dataset_stats()
    # with open("dataset_stats_train_users_train_data.picl","wb") as f:
    #     pickle.dump(photos_in_categories_train_train, f)

    # with open("dataset_stats_train_users_test_data.picl","wb") as f:
    #     pickle.dump(photos_in_categories_train_test, f)

    # print("Succesful retrieving train data")

    with open("dataset_stats_all_users_train_data.picl", "rb") as f:
        users_all_train = pickle.load(f)
        print(np.shape(test),test)
    with open("dataset_stats_all_users_test_data.picl", "rb") as f:
        users_all_test = pickle.load(f)

    

    #Statistics all user ()
    #Statistics train users 
    #Statistics global test users
    #Statistics 

    #We want the number of users with the different categories '
    #User with X categories (histogram), photos in each category.
    #We want the number of photos in each category
    #Average photos per user, average photo in each category, min photos in category, max photos in category