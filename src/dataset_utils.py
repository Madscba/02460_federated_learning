import pickle
import numpy as np
import os


def load_n_split(user, root_dir, train=True, train_proportion = 0.8, num_classes = None):
    user_f=user+'.pckl'
    f=open(os.path.join(root_dir,'data/img_lab_by_user',user_f,), 'rb')
    imgs = pickle.load(f)
    idx = int(train_proportion*len(imgs))
    if train:
        imgs=get_imgs_w_classes(np.array(imgs)[:idx],num_classes)
    else:
        imgs=relabel_imgs(np.array(imgs)[idx:])
    return imgs

def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90: # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)

def relabel_imgs(imgs):
    for i in range(len(imgs)):
        imgs[i,1] = relabel_class(imgs[i,1])
    return imgs

def get_imgs_w_classes(imgs,num_classes):
    imgs = relabel_imgs(imgs)
    if num_classes:
        all_classes=np.unique(imgs[:,1])
        classes=np.random.choice(all_classes,num_classes)
        if isinstance(classes,int):
            classes=np.array(classes)
        idx, _ = np.where(imgs[:,1]==classes[:,None])
        imgs=imgs[idx]
    return imgs
    

