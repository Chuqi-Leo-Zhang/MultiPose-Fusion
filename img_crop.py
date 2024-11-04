from skimage.io import imread, imsave
import numpy as np
import os


num_images = 16
image_size = 256

base_path = "/home/zhuominc/xlikeer_workspace/SyncDreamer/output/evaluation"
save_path = "/home/zhuominc/xlikeer_workspace/SyncDreamer/eval_GSO"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for image_dir in os.listdir(base_path):
    image_dir_path = os.path.join(base_path, image_dir)

    for image_file in os.listdir(image_dir_path):
        image_path = os.path.join(image_dir_path, image_file)
        img = imread(image_path)

        for index in range(num_images):
            rgb = np.copy(img[:, index*image_size : (index+1)*image_size, :])

            file_path = os.path.join(save_path, f"{image_dir}-pr")

            if not os.path.exists(file_path):
                os.makedirs(file_path)
            
            # print(os.path.join(file_path, f'{index:03}.png'))
            imsave(os.path.join(file_path, f'{index:03}.png'), rgb)
            