from check_data import run_check_data
from check_plucker import run_check_plucker
from delete_error_uids import run_delete_error_uids



if __name__ == "__main__":  

    error_num = 1
    base_path = "/Users/chuqizhang/Desktop/our_script/training"
    while error_num:
        run_check_data(base_path)
        print("Done checking data!")
        error_num = run_check_plucker(base_path)
        if error_num:
            print("Done checking plucker!")
            run_delete_error_uids(base_path)
            print("Had error but good now!")
        else: 
            print("Everything is fine!")        