from cgi import test
import numpy as np
import os
import matplotlib.pylab as plt
import pickle
import seaborn as sns
plt.style.use("seaborn")


# def generate_dummy_data(n=1000):
#     dummy_data = []
#     for user in range(n):
#         data_in_categories = np.random.randint(0,2,62)
#         dummy_data.append(data_in_categories)
#     print(dummy_data)
#     return dummy_data

def format_data_to_examples_in_each_category(datasets):
    data_category_totals = np.zeros((3, 62))
    for i in range(np.size(datasets)):
        for user_data in datasets[i]:
            data_category_totals[i, :] += user_data
    return data_category_totals


def format_data_to_users_with_each_category(datasets):
    """
    function that processses the dataset in order to find amount of users with each of the classes
    """
    users_with_category = np.zeros((np.size(datasets), 62))
    category_totals = np.zeros((np.size(datasets), 62))
    categories_per_user = [[] for i in range(np.size(datasets))]
    for i in range(np.size(datasets)):
        for user_data in datasets[i]:
            users_with_category[i, :] += np.array(user_data) > 0
            category_totals[i,:] += user_data
            categories_per_user[i].append(np.sum(np.array(user_data) > 0))
    return users_with_category, category_totals,categories_per_user


def visualize_category_distribution_over_classes(datasets, title='Histogram of X',
                                                 labels=['All', "Active", "Validation", "Test_local"]):
    kwargs = dict(alpha=0.6, bins=60)
    colors = ['black', 'red', 'blue']
    for i in range(np.size(datasets, 0)):
        print(i)
        plt.hist(datasets[i], **kwargs, color=colors[i], label=labels[i])
        # plt.hist(datasets[1], **kwargs, color='b', label='Train')
        # plt.hist(datasets[2], **kwargs, color='r', label='Test_global')
        # plt.hist(datasets[3], **kwargs, color='p', label='Test_local')

    plt.gca().set(title=title, ylabel='Frequency',xlabel="Class percentage")
    plt.xlim()
    plt.legend()
    plt.show()

def heatmap(dataset,title="",labels=["All_train","All_test","Train_train","Train_test","Validate_train","Validate_test"]):
    plt.figure(figsize=(10, 10))
    class_labels = [f"{i}" for i in range(64)]
    class_labels[62], class_labels[63] = "",""
    class_labels = np.asarray(class_labels).reshape(8,8)
    for i in range(np.size(dataset)):
        reshape_data = np.zeros(64)
        reshape_data[:62] = np.asarray(dataset[i])
        heat_map = sns.heatmap(reshape_data.reshape(8,8), linewidth=1, annot=class_labels,fmt="s")
        plt.title(title+" users: "+labels[i])
        plt.show()

if __name__ == "__main__":
    # all_users = generate_dummy_data(3597)
    # train_users = all_users[:3000]
    # test_users = all_users[3000:]

    with open("dataset_stats_all_users_train_data.picl", "rb") as f:
        users_all_train = pickle.load(f)
    with open("dataset_stats_all__users_test_data.picl", "rb") as f:
        users_all_test = pickle.load(f)

    with open("dataset_stats_test_users_train_data.picl", "rb") as f:
        users_test_train = pickle.load(f)
    with open("dataset_stats_test_users_test_data.picl", "rb") as f:
        users_test_test = pickle.load(f)

    with open("dataset_stats_train_users_train_data.picl", "rb") as f:
        users_train_train = pickle.load(f)
    with open("dataset_stats_train_users_test_data.picl", "rb") as f:
        users_train_test = pickle.load(f)

    ##### Visualize the category distribution from all users, test and train users.
    # photos_in_cat_all = np.sum(users_all_train, 0) + np.sum(users_all_test, 0)
    # photos_in_cat_train = np.sum(users_train_train, 0) + np.sum(users_train_test, 0)
    # photos_in_cat_test = np.sum(users_test_train, 0) + np.sum(users_test_test, 0)
    # photos = [photos_in_cat_all,photos_in_cat_train, photos_in_cat_test]
    # pct_photos = [photo/np.sum(photo) for photo in photos]
    # visualize_category_distribution_over_classes(pct_photos,title="Histogram of class distribution across data pools")
    ##### Print summary statistics for the same pools of data -> Used in a table
    # print([ [np.sum(photo),np.mean(photo), np.std(photo), np.min(photo), np.max(photo),np.median(photo)] for photo in photos])


    #MAKE A HEATMAP OF THE HOW MANY USERS WITH EACH CLASS AND EXAMPLES IN EACH (PRINT NUMBER INSIDE)
    datasets = np.array([users_all_train, users_all_test,users_train_train,users_train_test,users_test_train,users_test_test])
    users_with_category, category_totals,categories_per_user = format_data_to_users_with_each_category(datasets)

    heatmap(users_with_category,title="Amount of users with each class.")
    heatmap(category_totals,title="Amount of samples in each class")
    visualize_category_distribution_over_classes(categories_per_user)
    #MAKE HISTOGRAM OF CLASSES PER USER, EXAMPLES PER USER.
    np.sum([[user > 0] for user in users_all_train], 0)

    # Statistics all user ()
    # Statistics train users
    # Statistics global test users
    # Statistics

    # We want the number of users with the different categories '
    # User with X categories (histogram), photos in each category.
    # We want the number of photos in each category
    # Average photos per user, average photo in each category, min photos in category, max photos in category


    #visualize_category_distribution_over_classes(users_with_each_category_totals)
