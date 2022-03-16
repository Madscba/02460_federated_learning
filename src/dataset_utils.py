import pickle
import numpy as np
import os

def load_n_split(user, root_dir, train=True):
    user_f=user+'.pckl'
    f=open(os.path.join(root_dir,'data/img_lab_by_user',user_f,), 'rb')
    imgs = pickle.load(f)
    idx = int(0.8*len(imgs))
    if train:
        return np.array(imgs)[:idx]
    else:
        return np.array(imgs)[idx:]

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
