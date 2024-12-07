import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from ldm.base_utils import read_pickle, pose_inverse
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

from ldm.util import prepare_inputs
from spad.geometry import generate_batch


class SyncDreamerTrainData(Dataset):
    def __init__(self, target_dir, input_dir, uid_set_pkl, image_size=256):
        self.default_image_size = 256
        self.image_size = image_size
        self.target_dir = Path(target_dir)
        self.input_dir = Path(input_dir)

        self.uids = read_pickle(uid_set_pkl)

        print('============= length of dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16
        

    def __len__(self):
        return len(self.uids)

    def load_im(self, path):
        img = imread(path)
        img = img.astype(np.float32) / 255.0
        mask = img[:,:,3:]
        img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img, mask

    def process_im(self, im):
        im = im.convert("RGB")
        im = im.resize((self.image_size, self.image_size), resample=PIL.Image.BICUBIC)
        return self.image_transforms(im)

    def load_index(self, filename, index):
        img, _ = self.load_im(os.path.join(filename, '%03d.png' % index))
        img = self.process_im(img)
        return img

    def get_data_for_index(self, index):
        target_dir = os.path.join(self.target_dir, self.uids[index])
        input_dir = os.path.join(self.input_dir, self.uids[index])

        return {"target_image_dir": target_dir, "input_image_dir": input_dir}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data
    
    def collate_fn(self, batch):
        target_images_dir = [item['target_image_dir'] for item in batch]
        input_images_dir = [item['input_image_dir'] for item in batch]


        views = np.arange(0, self.num_images)
        
        start_view_index = np.random.randint(0, self.num_images) 
        views = (views + start_view_index) % self.num_images

        batch_target = []
        batch_input = []
        batch_input_elevation = []
        batch_input_azimuth = []
        batch_target_elevation = []
        batch_target_azimuth = []
        batch_target_poses = [] 

        for target_dir, input_dir in zip(target_images_dir, input_images_dir):
            target_images = []
            for si, target_index in enumerate(views):
                img = self.load_index(target_dir, target_index)
                target_images.append(img)
            
            target_images = torch.stack(target_images, 0)
            batch_target.append(target_images)
            input_img = self.load_index(input_dir, start_view_index)
            batch_input.append(input_img)

            K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(target_dir, f'meta.pkl'))
            target_elevation = torch.from_numpy(elevations.astype(np.float32))
            target_azimuth = torch.from_numpy(azimuths.astype(np.float32))
            target_poses = torch.from_numpy(cam_poses.astype(np.float32))
            batch_target_elevation.append(target_elevation) 
            batch_target_azimuth.append(target_azimuth)
            batch_target_poses.append(target_poses)


            K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_dir, f'meta.pkl'))
            input_elevation = torch.from_numpy(elevations[start_view_index:start_view_index+1].astype(np.float32))
            input_azimuth = torch.from_numpy(azimuths[start_view_index:start_view_index+1].astype(np.float32))
            batch_input_elevation.append(input_elevation)
            batch_input_azimuth.append(input_azimuth)
    
        return {"target_image": torch.stack(batch_target, 0), "input_image": torch.stack(batch_input, 0),\
                "input_elevation": torch.stack(batch_input_elevation, 0), "input_azimuth": torch.stack(batch_input_azimuth, 0),\
                "target_elevation": torch.stack(batch_target_elevation, 0), "target_azimuth": torch.stack(batch_target_azimuth, 0),\
                "target_poses": torch.stack(batch_target_poses, 0)}


class SyncDreamerEvalData(Dataset):
    def __init__(self, image_dir):
        self.image_size = 256
        self.image_dir = Path(image_dir)
        self.crop_size = 20

        self.fns = []
        for fn in Path(image_dir).iterdir():
            if fn.suffix=='.png':
                self.fns.append(fn)
        print('============= length of dataset %d =============' % len(self.fns))

    def __len__(self):
        return len(self.fns)

    def get_data_for_index(self, index):
        input_img_fn = self.fns[index]
        elevation = int(Path(input_img_fn).stem.split('-')[-1])
        # azimuth = int(Path(input_img_fn).stem.split('-')[-2])
        azimuth = 0
        return prepare_inputs(input_img_fn, elevation, azimuth, 200)

    def __getitem__(self, index):
        return self.get_data_for_index(index)

class SyncDreamerDataset(pl.LightningDataModule):
    def __init__(self, target_dir, input_dir, validation_dir, batch_size, uid_set_pkl, image_size=256, num_workers=4, seed=0, **kwargs):
        super().__init__()
        self.target_dir = target_dir
        self.input_dir = input_dir
        self.validation_dir = validation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uid_set_pkl = uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = SyncDreamerTrainData(self.target_dir, self.input_dir, uid_set_pkl=self.uid_set_pkl, image_size=256)
            self.val_dataset = SyncDreamerEvalData(image_dir=self.validation_dir)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler, collate_fn=self.train_dataset.collate_fn)
        # return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader

    def test_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
