from skimage.io import imread, imsave
import numpy as np
import os


num_images = 16
image_size = 256
view_num = 8

## single view
# base_path = "/home/zhuominc/xlikeer_workspace/SyncDreamer/output/evaluation"
# save_path = "/home/zhuominc/xlikeer_workspace/SyncDreamer/eval_GSO"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# for image_dir in os.listdir(base_path):
#     image_dir_path = os.path.join(base_path, image_dir)

#     for image_file in os.listdir(image_dir_path):
#         image_path = os.path.join(image_dir_path, image_file)
#         img = imread(image_path)

#         for index in range(0, num_images, view_num):
#             rgb = np.copy(img[:, index*image_size : (index + view_num)*image_size, :])

#             file_path = os.path.join(save_path, f"{image_dir}-pr")

#             if not os.path.exists(file_path):
#                 os.makedirs(file_path)
            
#             # print(os.path.join(file_path, f'{index:03}.png'))
#             imsave(os.path.join(file_path, f'{index:03}.png'), rgb)


base_path = "/home/zhuominc/leo/SyncDreamerCustomized/report_figs/multiview-16"
save_path = f"/home/zhuominc/leo/SyncDreamerCustomized/report_figs/multiview-{view_num}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for image_dir in os.listdir(base_path):
    image_path = os.path.join(base_path, image_dir)

    img = imread(image_path)

    for index in range(0, num_images, view_num):
        rgb = np.copy(img[:, index*image_size : (index + view_num)*image_size, :])
        
        imsave(os.path.join(save_path, f'{image_dir}-{index}-{index+view_num-1}.png'), rgb)
            