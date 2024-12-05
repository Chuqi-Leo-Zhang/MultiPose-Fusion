import os
import shutil
import pickle   


def run_check_data(base_path="training"):

    input_path = os.path.join(base_path, "input")
    input_dir_list = os.listdir(input_path)

    target_path = os.path.join(base_path, "target")
    target_path_list = os.listdir(target_path)

    print(len(input_dir_list))
    print(len(target_path_list))

    for name in input_dir_list:
        num_input_file = len(os.listdir(os.path.join(input_path, name)))
        if os.path.exists(os.path.join(target_path, name)):
            num_target_file = len(os.listdir(os.path.join(target_path, name)))
        else:
            print(name) 
            shutil.rmtree(os.path.join(input_path, name))
            continue    
        if num_input_file != 17 or num_target_file != 17:
            print(name)
            shutil.rmtree(os.path.join(input_path, name))
            shutil.rmtree(os.path.join(target_path, name))

    input_dir_list = os.listdir(input_path)
    target_path_list = os.listdir(target_path)
    for name in target_path_list:
        num_target_file = len(os.listdir(os.path.join(target_path, name)))
        if os.path.exists(os.path.join(input_path, name)):
            num_input_file = len(os.listdir(os.path.join(input_path, name)))
        else:
            print(name)
            shutil.rmtree(os.path.join(target_path, name))
            continue    

        if num_input_file != 17 or num_target_file != 17:
            print(name)
            shutil.rmtree(os.path.join(input_path, name))
            shutil.rmtree(os.path.join(target_path, name)) 


    assert sorted(os.listdir(input_path)) == sorted(os.listdir(target_path)), "The uids of input and target folders are not matched!"


    uids = os.listdir(input_path)
        
    def save_pickle(data, pkl_path):
        with open(pkl_path, 'wb+') as f:
            pickle.dump(data, f)

    print(len(uids))    
    save_pickle(uids, os.path.join(base_path, "uid_set.pkl"))


# scp -r /Users/chuqizhang/Desktop/our_script/empty zhuominc@lovelace.ece.local.cmu.edu:/home/zhuominc/leo/SyncDreamerCustomized/