from cgi import test
import numpy as np
import os
import matplotlib.pylab as plt



# def generate_dummy_data(n=1000):
#     dummy_data = []
#     for user in range(n):
#         data_in_categories = np.random.randint(0,2,62)
#         dummy_data.append(data_in_categories)
#     print(dummy_data)
#     return dummy_data

def format_data_to_examples_in_each_category(datasets):
    data_category_totals = np.zeros((3,62))
    for i in range(np.size(datasets)):
        for user_data in datasets[i]:
            data_category_totals[i,:] += user_data
    return data_category_totals

def format_data_to_users_with_each_category(datasets):
    data_category_totals = np.zeros((3,62))
    for i in range(np.size(datasets)):
        for user_data in datasets[i]:
            data_category_totals[i,:] += np.array(user_data)>0
    return data_category_totals

def visualize_category_distribution_over_classes(datasets,title='Histogram of X',labels= ['All',"Train","Test_global","Test_local"]):
    kwargs = dict(alpha=0.5, bins=100)
    colors = ['g','b','r','p']
    for i in range(np.size(datasets,0)):
        print(i)
        plt.hist(datasets[i], **kwargs, color=colors[i], label=labels[i])
        # plt.hist(datasets[1], **kwargs, color='b', label='Train')
        # plt.hist(datasets[2], **kwargs, color='r', label='Test_global')
        # plt.hist(datasets[3], **kwargs, color='p', label='Test_local')

    plt.gca().set(title=title, ylabel='Frequency')
    plt.xlim()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # all_users = generate_dummy_data(3597)
    # train_users = all_users[:3000]
    # test_users = all_users[3000:]


    with open("dataset_stats_all_users_train_data.picl", "rb") as f:
        users_all_train = pickle.load(f)
        print(np.shape(test),test)
    with open("dataset_stats_all_users_test_data.picl", "rb") as f:
        users_all_test = pickle.load(f)

    
    print(f"We have {len(users_all_train)} users in total that combined have {np.sum(np.sum(users_all_train)+np.sum(users_all_test))} handwritten digits")
    print(f"On average each user has ")
    #Statistics all user ()
    #Statistics train users 
    #Statistics global test users
    #Statistics 

    #We want the number of users with the different categories '
    #User with X categories (histogram), photos in each category.
    #We want the number of photos in each category
    #Average photos per user, average photo in each category, min photos in category, max photos in category
    datasets = np.array([users_all_train,users_all_test])   
    examples_in_each_category_totals = format_data_to_examples_in_each_category(datasets)
    users_with_each_category_totals = format_data_to_users_with_each_category(datasets)
    print(users_with_each_category_totals[0])
    visualize_category_distribution_over_classes(users_with_each_category_totals)

