import pickle
import numpy as np

def dump_pkl(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def chk_pkl(file_name):
    data = load_pickle(file_name)
    print(data.shape)
    print(data) 
