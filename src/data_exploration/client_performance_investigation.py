import os
import sys
sys.path.append(r"C:\Users\Mads-\Documents\Universitet\Kandidat\2. Semester\02460_advanced_ml\02460_federated_learning\src")
sys.path.append(r"C:\Users\Mads-\Documents\Universitet\Kandidat\2. Semester\02460_advanced_ml\02460_federated_learning\src\data_exploration")
from global_model_eval import global_model_eval
from data_exploration import visualize_category_distribution_over_classes 
import numpy as np
import time
from model import Net
import torch
import matplotlib.pylab as plt


def visualize_data(x_data,y_data,model,client_names):
    if get_loss:
        loss_func = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in zip(x_data, y_data):
            x = torch.load(x).to(DEVICE)
            y = torch.load(y).to(DEVICE)
            num_obs_per_user.append(x.shape[0])

            # batch_size = 8
            # use all the data instead of an actual batch size
            batch_size = num_obs_per_user[-1]
            for i in range(num_obs_per_user[-1] // batch_size):
                x_ = x[i * batch_size:i * batch_size + batch_size]
                y_ = y[i * batch_size:i * batch_size + batch_size]

                pred = net(x_)
                acc.append(torch.mean((torch.argmax(pred, axis=1) == y_).type(torch.float)).item() * 100)

                plot_8x8(x,torch.argmax(pred, axis=1),y_,acc[-1],client_names[i])
                # if get_loss:
                #     loss.append(loss_func(pred, y_).item())
def plot_8x8(classes,preds,labels,overall_acc,client_name):
    plt.figure(figsize=(15, 15))
    #reshaped_img = classes[i].reshape(128,128)
    num_row = 8
    num_col = 8

    # get a segment of the dataset
    num = num_row * num_col
    images = [class_.reshape(128,128) for class_ in classes]
    category_labels = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","",""]

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2.5 * num_row))
    # plot images
    for i, (pred,label) in enumerate(zip(preds,labels)):
        if i % 64 == 0 and i > 0:
            plt.tight_layout()
            fig.suptitle("User: {} , Overall accuracy: {}".format(client_name[:-3],np.round(overall_acc,1)))
            plt.show()
            if not i/64 >= len(preds)//64:
                fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2.5 * num_row))

        if i/64 >= len(preds)//64:
            break
        ax = axes[(i%64) // num_col, (i%64) % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}, Pred: {}'.format(category_labels[label],category_labels[pred]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


if __name__ == "__main__":
    state_dict = r"C:\Users\Mads-\Documents\Universitet\Kandidat\2. Semester\02460_advanced_ml\02460_federated_learning\saved_models\Fedavg_state_dict_16_15_48.pt"
    data_folder = r"C:\Users\Mads-\Documents\Universitet\Kandidat\2. Semester\02460_advanced_ml\02460_federated_learning\dataset\test_stored_as_tensors"
    num_test_clients = 20 # i.e. fetch all clients
    get_loss = True

    client_names = os.listdir(data_folder)
    x_data = np.array(sorted([os.path.join(data_folder, client) for client in client_names if client.endswith("x.pt")])[:num_test_clients])
    y_data = np.array(sorted([os.path.join(data_folder, client) for client in client_names if client.endswith("y.pt")])[:num_test_clients])

    acc, loss, num_obs_per_user = global_model_eval(state_dict=state_dict,data_folder= data_folder, num_test_clients=num_test_clients, get_loss=get_loss)


    no_clients = 2
    min_clients = np.argsort(acc)[:no_clients] #Pick 10 clients with lowest accuracy
    max_clients = np.argsort(acc)[-no_clients:] #Pick 10 clients with highest accuracy

    visualize_category_distribution_over_classes([acc], title='Histogram of client accuracies (Validation clients)',labels=["Validation"],xlabel="Client accuracies",ylabel="Amount of clients / Frequency")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.load_state_dict(torch.load(state_dict))

    visualize_data(x_data=x_data[min_clients],y_data=y_data[min_clients],model=net,client_names=np.array(client_names)[min_clients])
    visualize_data(x_data=x_data[max_clients],y_data=y_data[max_clients],model=net,client_names=np.array(client_names)[max_clients])

    print("mean_acc:", np.mean(np.asarray(acc)))
    print(loss)
    print(num_obs_per_user)

