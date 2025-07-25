import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
import webdataset as wds

from ..utils.misc import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    

class ShapeNet(Dataset):
    '''
    statistics: {
        '02691156': 4045, '02747177': 343, '02773838': 83, '02801938': 113, 
        '02808440': 856, '02818832': 233, '02828884': 1813, '02843684': 73, 
        '02871439': 452, '02876657': 498, '02880940': 186, '02924116': 939, 
        '02933112': 1571, '02942699': 113, '02946921': 108, '02954340': 56, 
        '02958343': 3514, '02992529': 831, '03001627': 6778, '03046257': 651, 
        '03085013': 65, '03207941': 93, '03211117': 1093, '03261776': 73, 
        '03325088': 744, '03337140': 298, '03467517': 797, '03513137': 162, 
        '03593526': 596, '03624134': 424, '03636649': 2318, '03642806': 460, 
        '03691459': 1597, '03710193': 94, '03759954': 67, '03761084': 152, 
        '03790512': 337, '03797390': 214, '03928116': 239, '03938244': 96, 
        '03948459': 307, '03991062': 602, '04004475': 166, '04074963': 66, 
        '04090263': 2373, '04099429': 85, '04225987': 152, '04256520': 3173, 
        '04330267': 218, '04379243': 8436, '04401088': 1089, '04460130': 133, 
        '04468005': 389, '04530566': 1939, '04554684': 169
    }
    **********************************************************************************
    category number:  55
    sample number:  52472
    max: 04379243, 8436
    min: 02954340, 56
    **********************************************************************************
    surface: dict, keys=[points, normals, loc, scale], (100000, 3), (100000, 3)
    ShapeNetV2_point: dict, key=[vol_points, vol_label, near_points, near_label], (500000, 3), (500000,), (500000, 3), (500000,)
    '''
    def __init__(self, dataset_folder, split, categories=['03001627'], sampling=True, num_samples=4096, surface_sampling=True, pc_size=8192, rotate_prob=0.2):

        self.pc_size = pc_size
        self.rotate_prob = rotate_prob

        if split == 'train':
            self.transform = AxisScaling((0.75, 1.25), True)
            self.shape_transform = ShapeAugTransform()
        else:
            self.transform = None
            self.shape_transform = None

        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        self.mesh_folder = os.path.join(self.dataset_folder, 'surface')

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]

        categories.sort()
        # print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]
        print('============= length of {} dataset: {} ============='.format(split, len(self.models)))

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        
        point_path = os.path.join(self.point_folder, category, model+'.npz')
        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(point_path)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()
        
        pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model+'.npz')
        with np.load(pc_path) as data:
            points = data['points'].astype(np.float32)
            normals = data['normals'].astype(np.float32)
            surface = np.concatenate([points, normals], axis=-1)
            surface = surface * scale

        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
            surface = surface[ind]
        surface = torch.from_numpy(surface)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)
    
        data = {
            'surface': surface,
            'geo_points': torch.cat([points, labels.unsqueeze(-1)], dim=-1),
            'uid': f'{category}_{model}'
        }

        if self.shape_transform:
            data = self.shape_transform(data)

        return data
        

    def __len__(self):
        return len(self.models)
        

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        surface_points = surface[..., :3]
        surface_normals = surface[..., 3:]

        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface_points = surface_points * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface_points).max().item()) * 0.999999
        surface_points *= scale
        point *= scale

        if self.jitter:
            surface_points += 0.005 * torch.randn_like(surface_points)
            surface_points.clamp_(min=-1, max=1)
        
        surface = torch.cat([surface_points, surface_normals], dim=-1).float()

        return surface, point


class ShapeAugTransform(object):
    def __init__(self,
            scale_interval=(0.8, 1.),
            rot_fix_axis=False,
            prob=0.2,
            ) -> None:
        self.scale_interval = scale_interval
        self.rot_fix_axis = rot_fix_axis
        self.prob = prob

    def __call__(self, sample):
        surface_points = sample["surface"][..., :3]
        surface_normals = sample["surface"][..., 3:]
        geo_points = sample["geo_points"][..., :3]
        geo_labels = sample["geo_points"][..., 3:]
        
        if random.random() <= self.prob / 2:
            # 1. aug scale
            scaling = random.uniform(self.scale_interval[0], self.scale_interval[1])
            surface_points = surface_points * scaling
            geo_points = geo_points * scaling

        elif random.random() <= self.prob:
            # 2. aug rotation
            if self.rot_fix_axis:
                raise NotImplementedError
                # rotation_angle = None
                # rotation_matrix = torch.as_tensor(
                #     [
                #         [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                #         [0, 1, 0],
                #         [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
                #     ]
                # )
            else:
                rotation_matrix = np.random.rand(3, 3)
                rotation_matrix, _ = np.linalg.qr(rotation_matrix)

            surface_points = surface_points @ rotation_matrix.T
            surface_normals = surface_normals @ rotation_matrix.T
            geo_points[..., :3] = geo_points[..., :3] @ rotation_matrix.T

            # fix scale
            scale = (1 / torch.abs(surface_points).max().item()) * 0.999999
            surface_points = surface_points * scale
            geo_points = geo_points * scale

        sample["surface"] = torch.cat([surface_points, surface_normals], dim=-1).float()
        sample["geo_points"] = torch.cat([geo_points, geo_labels], dim=-1).float()

        return sample