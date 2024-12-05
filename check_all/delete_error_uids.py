import pickle
import os
import shutil
from tqdm import tqdm


def read_pickle_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)    
    
def write_pickle_file(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def run_delete_error_uids(base_path="training"):    

    all_uids = read_pickle_file(os.path.join(base_path, 'uid_set.pkl'))
    error_uids = read_pickle_file(os.path.join(base_path, 'error_uids.pkl'))
    print('Before delete, all uids:', len(all_uids)) 
    print('error_uids:', len(error_uids))

    input_path = os.path.join(base_path, 'input')
    target_path = os.path.join(base_path, 'target') 
    for uid in tqdm(all_uids):
        if uid in error_uids:
            # print(uid)
            shutil.rmtree(os.path.join(input_path, uid))    
            shutil.rmtree(os.path.join(target_path, uid))
            all_uids.remove(uid)

    write_pickle_file(os.path.join(base_path, 'uid_set.pkl'), all_uids)
    os.remove(os.path.join(base_path, 'error_uids.pkl'))