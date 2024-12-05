import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import PIL.Image as Image
import pickle
from torch.utils.data import Dataset
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange


import torchvision.transforms as transforms
import torchvision

from pytorch_lightning import seed_everything
from tqdm import tqdm

seed_everything(6033)
np.seterr(invalid='raise')  


class SyncDreamerTrainData(Dataset):
    def __init__(self, target_dir, input_dir, uid_set_pkl, image_size=256):
        self.default_image_size = 256
        self.image_size = image_size
        self.target_dir = Path(target_dir)
        self.input_dir = Path(input_dir)

        self.uids = read_pickle(uid_set_pkl)
        print('============= length of dataset %d =============' % len(self.uids))

        self.num_images = 16

    def __len__(self):
        return len(self.uids)


    def load_index(self, filename, index):
        img, _ = self.load_im(os.path.join(filename, '%03d.png' % index))
        img = self.process_im(img)
        return img

    def get_data_for_index(self, index):
        target_dir = os.path.join(self.target_dir, self.uids[index])
        input_dir = os.path.join(self.input_dir, self.uids[index])

        return {"target_image_dir": target_dir, "input_image_dir": input_dir, "uids": self.uids[index]}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data
    
    def collate_fn(self, batch):
        target_images_dir = [item['target_image_dir'] for item in batch]
        input_images_dir = [item['input_image_dir'] for item in batch]
        uids = [item['uids'] for item in batch] 
        
        views = np.arange(0, self.num_images)
        start_view_index = np.random.randint(0, self.num_images)
        views = (views + start_view_index) % self.num_images

        batch_target = []
        batch_input = []
        batch_input_elevation = []
        batch_input_azimuth = []
        batch_target_elevation = []
        batch_target_azimuth = []
        # input_elevation = None
        # input_azimuth = None
        # target_elevation = None
        # target_azimuth = None
        for target_dir, input_dir in zip(target_images_dir, input_images_dir):
            target_images = []


            # if target_elevation is None and target_azimuth is None:
            K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(target_dir, f'meta.pkl'))
            target_elevation = torch.from_numpy(elevations.astype(np.float32))
            target_azimuth = torch.from_numpy(azimuths.astype(np.float32))
            batch_target_elevation.append(target_elevation) 
            batch_target_azimuth.append(target_azimuth) 

            # if input_elevation is None and input_azimuth is None:
            K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_dir, f'meta.pkl'))
            input_elevation = torch.from_numpy(elevations[start_view_index:start_view_index+1].astype(np.float32))
            input_azimuth = torch.from_numpy(azimuths[start_view_index:start_view_index+1].astype(np.float32))
            batch_input_elevation.append(input_elevation)
            batch_input_azimuth.append(input_azimuth)

        # return {"target_image": torch.stack(batch_target, 0), "input_image": torch.stack(batch_input, 0),\
        #         "input_elevation": torch.stack(batch_input_elevation, 0), "input_azimuth": torch.stack(batch_input_azimuth, 0),\
        #         "target_elevation": target_elevation, "target_azimuth": target_azimuth}


    
        return {"input_elevation": torch.stack(batch_input_elevation, 0), "input_azimuth": torch.stack(batch_input_azimuth, 0),\
                "target_elevation": torch.stack(batch_target_elevation, 0), "target_azimuth": torch.stack(batch_target_azimuth, 0),\
                "uids": uids}
    
def prepare(batch):


        elevation_input = batch['input_elevation'][:, 0] # b
        azimuth_input = batch['input_azimuth'][:, 0] # b
        elevation_output = batch['target_elevation'][:,:] # B,N
        azimuth_output = batch['target_azimuth'][:,:] # B,N
        input_info = {'input_elevation': elevation_input,
                      'input_azimuth': azimuth_input,
                      'target_elevation': elevation_output,
                      'target_azimuth': azimuth_output,
                      'uids': batch['uids']}

        return input_info

def generate_batch(elevations=[45,45,45,45], azimuths=[0,90,180,270], use_abs=False):

    # assert first frame is identity
    batch = get_batch_from_spherical(elevations, azimuths)
    return True

def get_batch_from_spherical(elevations, azimuths, fov=0.8575560548920328, image_size=256):
    """Given a list of elevations and azimuths, generates cameras, computes epipolar masks and plucker embeddings and organizes them as a batch."""

    num_views = len(elevations)
    latent_size = image_size // 8
    assert len(elevations) == len(azimuths)

    # intialize all epipolar masks to ones (i.e. all pixels are considered)
    batch_attention_masks = torch.ones(num_views, num_views, latent_size ** 2, latent_size ** 2, dtype=torch.bool)
    plucker_embeds = [None for _ in range(num_views)]

    # compute pairwise mask and plucker
    for i, icam in enumerate(zip(elevations, azimuths)):
        for j, jcam in enumerate(zip(elevations, azimuths)):
            if i == j: continue

            first_frame = edict({"fov": fov}); second_frame = edict({"fov": fov})
            first_frame["camera"] = get_blender_from_spherical(elevation=icam[0], azimuth=icam[1])
            second_frame["camera"] = get_blender_from_spherical(elevation=jcam[0], azimuth=jcam[1])

            continue
    return True

def get_blender_from_spherical(elevation, azimuth):
    """ Generates blender camera from spherical coordinates. """

    cartesian_coords = spherical_to_cartesian(np.array([[elevation, azimuth, 1.5]]))
    
    # get camera rotation
    center = np.array([0, 0, 0])
    eye = cartesian_coords[0]
    up = np.array([0, 0, 1])

    R, T = look_at(eye, center, up)

    return True

def look_at(eye, center, up):
    # Create a normalized direction vector from eye to center
    f = np.array(center) - np.array(eye)
    f /= np.linalg.norm(f)

    # Create a normalized right vector
    up_norm = np.array(up) / np.linalg.norm(up)
    s = np.cross(f, up_norm)
    s /= (np.linalg.norm(s))

    return True, True

def spherical_to_cartesian(spherical_coords):
    # convert from spherical to cartesian coordinates
    theta, azimuth, radius = spherical_coords.T
    x = radius * np.sin(theta) * np.cos(azimuth)
    y = radius * np.sin(theta) * np.sin(azimuth)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

def save_pickle(data, pkl_path):
    with open(pkl_path, 'wb+') as f:
        pickle.dump(data, f)

def read_pickle(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def run_check_plucker(base_path="training"):    

    input_path = os.path.join(base_path, "input")   
    target_path = os.path.join(base_path, "target")
    uid_path = os.path.join(base_path, "uid_set.pkl")

    train_dataset = SyncDreamerTrainData(target_dir=target_path, input_dir=input_path, uid_set_pkl=uid_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=train_dataset.collate_fn)

    error_uids = []
    for data in tqdm(train_loader):

        input_info = prepare(data)
        N = 16
        B = 1
        target_index = torch.randint(0, N, (B, 1)).long() # B, 1
        input_elev, target_elev = input_info['input_elevation'].to("cpu"), input_info['target_elevation'][torch.arange(B)[:,None], target_index][:,0].to("cpu")
        input_azim, target_azim = input_info['input_azimuth'].to("cpu"), input_info['target_azimuth'][torch.arange(B)[:,None], target_index][:,0].to("cpu")
        
        att_masks, plucker_embeds = [], []
        for i in range(B):
            elevations = [target_elev[i], input_elev[i]]
            azimuths = [target_azim[i], input_azim[i]]
            try:
                plc_batch = generate_batch(elevations, azimuths)
            except:
                error_uids += input_info['uids']

    print(len(error_uids))
    if len(error_uids) > 0:
        save_pickle(error_uids, os.path.join(base_path, "error_uids.pkl"))
        return len(error_uids)
    else:
        return 0