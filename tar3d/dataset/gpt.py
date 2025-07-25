import os
import json
import numpy as np
import random
from tqdm import tqdm
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset
from .utils import AxisScaling, ShapeAugTransform


bg_color = [1., 1., 1.]



class UlipTextTo3DDataset(Dataset):
    def __init__(self, args=None, transform=None):
        dataset_folder = '/cpfs01/user/zhangxuying/Datasets/ShapeNetV2'
        self.dataset_folder = dataset_folder

        shapenet_uid2cond_file=f'{dataset_folder}/uid2conds.json'
        ulip_root='/cpfs01/user/zhangxuying/Datasets/ULIP'
        self.ulip_root = ulip_root

        # prompts
        assert os.path.isfile(shapenet_uid2cond_file)
        with open(shapenet_uid2cond_file, 'r') as f:
            uid2cond_dict = json.load(f)
        '''
        {
            uid: [
                (image_path, text_captions), 
            ],
        }
        '''

        self.uid2cond_dict = {uid: [pair[1] for pair in img_text_pairs] for uid, img_text_pairs in uid2cond_dict.items()}

        
        # 3D info
        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        self.mesh_folder = os.path.join(self.dataset_folder, 'surface')

        categories = os.listdir(self.point_folder)
        categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]

        categories.sort()
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, 'train.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        print('============= length of dataset: {} ============='.format(len(self.models)))        
      
        # self.transform = transform
        self.transform = AxisScaling((0.75, 1.25), True)
        self.shape_transform = ShapeAugTransform()

        self.is_scale=True
        self.surface_sampling=True
        self.pc_size=81920
        self.num_samples = 20480
        self.sampling = True

        latent_size = 32
        self.code_len = 3* (latent_size ** 2)
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.models)

    def read_cap_feat(self, uid, view_id):

        '''
        02691156_10af5de930178a161596c26b5af806fe
        '''
        file_path = os.path.join(self.ulip_root, 'captions/t5_ulip_feats', uid+'.npz')
        assert os.path.isfile(file_path)
        cap_feats = np.load(
            file_path,
            allow_pickle=True
        )
        view_id = sorted(list(cap_feats.keys()))[view_id]
        # print(type(view_id), view_id)
        t5_feat = torch.from_numpy(cap_feats[view_id])
        return t5_feat

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
            if self.is_scale:
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

        near_points = torch.from_numpy(near_points)
        near_label = torch.from_numpy(near_label).float()
        points = torch.cat([vol_points, near_points], dim=0)
        labels = torch.cat([vol_label, near_label], dim=0)

        if self.transform:
            surface, points = self.transform(surface, points)

        data = {
            'surface': surface,
            'geo_points': torch.cat([points, labels.unsqueeze(-1)], dim=-1),
            'uid': f'{category}_{model}'
        }

        if self.shape_transform:
            data = self.shape_transform(data)
        surface = data['surface']

        uid = self.models[idx]['category'] + '_' + self.models[idx]['model']
        view_id = random.randint(0, 3)
        try:
            t5_feat = self.read_cap_feat(uid, view_id)
            valid = 1
        except:
            t5_feat = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
            valid = 0

        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        t5_feat_len = t5_feat.shape[1] 
        feat_len = min(self.t5_feature_max_len, t5_feat_len)
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        emb_mask = torch.zeros((self.t5_feature_max_len,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
        T = self.t5_feature_max_len
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)

        return surface, t5_feat_padding, attn_mask, torch.tensor(valid)


class UlipImageTo3DDataset(Dataset):
    def __init__(self, args=None, transform=None):
        '''background color, default: white'''
        bg_white = [1., 1., 1.]
        bg_black = [0., 0., 0.]


        dataset_folder = '/cpfs01/user/zhangxuying/Datasets/ShapeNetV2'
        self.dataset_folder = dataset_folder

        # prompt
        shapenet_uid2cond_file=f'{dataset_folder}/uid2conds.json'
        ulip_root='/cpfs01/shared/public/lumina/Datasets/ULIP'
        self.ulip_root = ulip_root
      
        assert os.path.isfile(shapenet_uid2cond_file)
        with open(shapenet_uid2cond_file, 'r') as f:
            uid2cond_dict = json.load(f)
        '''
        {
            uid: [
                (image, text_captions), 
            ],
        }
        '''

        self.uid2cond_dict = {uid: [pair[0] for pair in img_text_pairs] for uid, img_text_pairs in uid2cond_dict.items()}

        
        # 3D info
        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        self.mesh_folder = os.path.join(self.dataset_folder, 'surface')

        categories = os.listdir(self.point_folder)
        categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]

        categories.sort()
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.point_folder, c)
            assert os.path.isdir(subpath)

            split_file = os.path.join(subpath, 'train.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in models_c
            ]

        print('============= length of dataset: {} ============='.format(len(self.models)))        
      

        self.transform = AxisScaling((0.75, 1.25), True)
        self.shape_transform = ShapeAugTransform()

        self.is_scale=True
        self.surface_sampling=True
        self.pc_size=81920
        self.num_samples = 20480
        self.sampling = True

        latent_size = 32
        self.code_len = 3* (latent_size ** 2)
        self.input_img_size = 224
        self.dino_feature_max_len = 197
        self.t5_feature_dim = 768
        self.max_seq_length = self.dino_feature_max_len + self.code_len

    def __len__(self):
        return len(self.models)

    def read_dino_feat_from_img_path(self, uid, view_id, which_data="Shapenet"):
        file_path = os.path.join(self.ulip_root, 'dino_feats', which_data, uid+'.npz')
        assert os.path.isfile(file_path)
        cap_feats = np.load(
            file_path,
            allow_pickle=True
        )
        view_id = sorted(list(cap_feats.keys()))[view_id]
        dino_feats = torch.from_numpy(cap_feats[view_id])
        return dino_feats


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
            if self.is_scale:
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

        near_points = torch.from_numpy(near_points)
        near_label = torch.from_numpy(near_label).float()
        points = torch.cat([vol_points, near_points], dim=0)
        labels = torch.cat([vol_label, near_label], dim=0)

        if self.transform:
            surface, points = self.transform(surface, points)

        data = {
            'surface': surface,
            'geo_points': torch.cat([points, labels.unsqueeze(-1)], dim=-1),
            'uid': f'{category}_{model}'
        }

        if self.shape_transform:
            data = self.shape_transform(data)
        surface = data['surface']

        view_id = random.randint(0, 3)
        try:
            uid = f'{category}_{model}'
            img_feat = self.read_dino_feat_from_img_path(uid, view_id, which_data='Shapenet')
            valid = 1
        except:
            valid = 0
            img_feat = torch.zeros(1, self.dino_feature_max_len, self.t5_feature_dim)

        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))

        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)

        return surface, img_feat, attn_mask, torch.tensor(valid)




def build_uliptTo3D(args=None, transform=None):
    return UlipTextTo3DDataset(args, transform)


def build_ulipiTo3D(args=None, transform=None):
    return UlipImageTo3DDataset(args, transform)
