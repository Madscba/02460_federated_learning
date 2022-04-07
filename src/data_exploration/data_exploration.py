from cgi import test
import numpy as np
import os
import matplotlib.pylab as plt
from matplotlib import pyplot
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

def heatmap(dataset,title="",class_labels=[f"{i}" for i in range(64)]):
    class_labels[62], class_labels[63] = "",""
    class_labels = np.asarray(class_labels).reshape(8,8)
    reshape_data = np.zeros(64)
    reshape_data[:62] = np.asarray(dataset)
    heat_map = sns.heatmap(reshape_data.reshape(8,8), linewidth=1, annot=class_labels,fmt="s",xticklabels=False,yticklabels=False)
    plt.title(title)
    plt.show()
def plot_classes(classes):
    plt.figure(figsize=(15, 15))
    #reshaped_img = classes[i].reshape(128,128)
    num_row = 8
    num_col = 8

    # get a segment of the dataset
    num = num_row * num_col
    images = [class_.reshape(128,128) for class_ in classes]
    labels = list(range(62))

    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(62):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # all_users = generate_dummy_data(3597)
    # train_users = all_users[:3000]
    # test_users = all_users[3000:]
    #
    ## The folowing file names users_<
    with open("dataset_stats_all_users_train_data.picl", "rb") as f:
        users_all_train = pickle.load(f)
    with open("dataset_stats_all_users_test_data.picl", "rb") as f:
        users_all_test = pickle.load(f)
    with open("dataset_stats_all_users_all_data.picl", "rb") as f:
        users_all_all = pickle.load(f)

    with open("dataset_stats_test_users_train_data.picl", "rb") as f:
        users_test_train = pickle.load(f)
    with open("dataset_stats_test_users_test_data.picl", "rb") as f:
        users_test_test = pickle.load(f)
    with open("dataset_stats_test_users_all_data.picl", "rb") as f:
        users_test_all = pickle.load(f)

    with open("dataset_stats_train_users_train_data.picl", "rb") as f:
        users_train_train = pickle.load(f)
    with open("dataset_stats_train_users_test_data.picl", "rb") as f:
        users_train_test = pickle.load(f)
    with open("dataset_stats_train_users_all_data.picl", "rb") as f:
        users_train_all = pickle.load(f)


    #### PLOT CATEGORY EXAMPLES
    # with open("category_examples_test.picl", "rb") as f:
    #     category_examples_all = pickle.load(f)
    # plot_classes(category_examples_all)

    ##### Visualize the category distribution from all users, test and train users.
    # photos_in_cat_all = np.sum(users_all_all, 0)
    # photos_in_cat_train = np.sum(users_train_all, 0)
    # photos_in_cat_test = np.sum(users_test_all, 0)
    # photos = [photos_in_cat_all,photos_in_cat_train, photos_in_cat_test]
    # pct_photos = [photo/np.sum(photo) for photo in photos]
    # visualize_category_distribution_over_classes(pct_photos,title="Histogram of class distribution across data pools")
    ##### Print summary statistics for the same pools of data -> Used in a table
    # print([ [np.sum(photo),np.mean(photo), np.std(photo), np.min(photo), np.max(photo),np.median(photo)] for photo in photos])


    category_labels = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","",""]
    #MAKE A HEATMAP OF THE HOW MANY USERS WITH EACH CLASS AND EXAMPLES IN EACH (PRINT NUMBER INSIDE)
    datasets = np.array([users_all_train, users_all_test,users_all_all,users_train_train,users_train_test,users_train_all,users_test_train,users_test_test,users_test_all])
    users_with_category, category_totals,categories_per_user = format_data_to_users_with_each_category(datasets)

    #["All data","Active users train data","Active users test data","Validatation data"]
    #INVESTIGATE HOW THE CATEGORIES ARE SPREAD OUT AMONGST THE USERS
    heatmap( ((users_with_category[2] ) / (np.sum( users_with_category[2] )))*100,title="Percentage of users with each class. All data",class_labels=category_labels)
    heatmap( (users_with_category[3] / np.sum(users_with_category[3]))*100,title="Percentage of users with each class training data (active clients)",class_labels=category_labels)
    heatmap( (users_with_category[4] / np.sum(users_with_category[4]))*100,title="Percentage of users with each class test data (active clients)",class_labels=category_labels)
    heatmap( ((users_with_category[8] ) / np.sum(users_with_category[8]))*100,title="Percentage of users with each class. Test data",class_labels=category_labels)
    heatmap( 100*abs( ( (users_with_category[3] / np.sum(users_with_category[3]) )- (users_with_category[4] / np.sum(users_with_category[4])) ) ),title="Difference in user percentages between active clients train/test",class_labels=category_labels)

    #INVESTIGATE HOW THE AMOUNT OF EXAMPLES IS DISTRIBUTED OVER THE CLASSES
    heatmap(((category_totals[2]) / (np.sum(category_totals[2]))) * 100,
            title="Percentage of samples within each class. All data", class_labels=category_labels)
    heatmap((category_totals[3] / np.sum(category_totals[3])) * 100,
            title="Percentage of samples within each class training data (active clients)", class_labels=category_labels)
    heatmap((category_totals[4] / np.sum(category_totals[4])) * 100,
            title="Percentage of samples within each class test data (active clients)", class_labels=category_labels)
    heatmap(((category_totals[8]) / np.sum(category_totals[8])) * 100,
            title="Percentage of samples within each class. Test data", class_labels=category_labels)
    heatmap(100 * abs(((category_totals[3] / np.sum(category_totals[3])) - (
                category_totals[4] / np.sum(category_totals[4])))),
            title="Difference in sample percentages across classes between active clients train/test", class_labels=category_labels)

    visualize_category_distribution_over_classes([categories_per_user[2]/np.sum(categories_per_user[2]),categories_per_user[5]/np.sum(categories_per_user[5]),categories_per_user[8]/np.sum(categories_per_user[8])], title="Histogram of classes per user across data pools")
    print("A")
    #visualize_category_distribution_over_classes(categories_per_user)
    #MAKE HISTOGRAM OF CLASSES PER USER, EXAMPLES PER USER.
    # np.sum([[user > 0] for user in users_all_train], 0)

    # Statistics all user ()
    # Statistics train users
    # Statistics global test users
    # Statistics

    # We want the number of users with the different categories '
    # User with X categories (histogram), photos in each category.
    # We want the number of photos in each category
    # Average photos per user, average photo in each category, min photos in category, max photos in category


    #visualize_category_distribution_over_classes(users_with_each_category_totals)
