import pickle
import os

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

path = os.path.join(parent_path, 'data', 'intermediate', 'images_by_writer.pkl')
f=open(path, 'rb')
pckl_obj=pickle.load(f)
os.chdir(os.path.join(parent_path, 'data','img_lab_by_user'))
for tup in pckl_obj:
    fn=f'{tup[0]}.pckl'
    file_pi = open(fn, 'wb') 
    pickle.dump(tup[1], file_pi)