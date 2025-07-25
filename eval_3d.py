import os
import json
import numpy as np
import trimesh
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tar3d.evaluation.chamfer_distance import ChamferDistance

from tar3d.utils import load_mesh
from .eval_2d import transform_mesh


def measure3d(mesh_gt_path, mesh_pred_path, method='instantmesh', num_pts=16000, f_thd=0.02):
    chamfer_distance = ChamferDistance()

    # load predicted_result
    mesh_pred = load_mesh(mesh_pred_path)
    mesh_pred = transform_mesh(mesh_pred, method_name=method)
    mesh_pred.fix_normals()

    points_pred, face_indices_pred = trimesh.sample.sample_surface(mesh_pred, num_pts)
    points_pred = torch.from_numpy(points_pred).float().to('cuda')
    normals_pred = torch.from_numpy(mesh_pred.face_normals[face_indices_pred]).float().to('cuda')
    print(points_pred.shape)

    # load gt
    mesh_gt = load_mesh(mesh_gt_path)
    mesh_gt = transform_mesh(mesh_gt, method_name='gt')
    mesh_gt.fix_normals()

    points_gt, face_indices_gt = trimesh.sample.sample_surface(mesh_gt, num_pts)
    points_gt = torch.from_numpy(points_gt).float().to('cuda')
    normals_gt = torch.from_numpy(mesh_gt.face_normals[face_indices_gt]).float().to('cuda')
    print(points_gt.shape)

    dist1, dist2, idx1, idx2 = chamfer_distance(points_pred.unsqueeze(0), points_gt.unsqueeze(0))

    dist1 = dist1.squeeze(0)
    dist2 = dist2.squeeze(0)
    idx1 = idx1.squeeze(0)
    idx2 = idx2.squeeze(0)

    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    pred_to_gt_chamfer = torch.mean(dist1).item()
    gt_to_pred_chamfer = torch.mean(dist2).item()
    chamfer = (pred_to_gt_chamfer + gt_to_pred_chamfer) / 2

    # check idx
    dist1_ = F.mse_loss(points_pred, points_gt[idx1.long()], reduction='none').sum(dim=-1).sqrt()
    dist2_ = F.mse_loss(points_gt, points_pred[idx2.long()], reduction='none').sum(dim=-1).sqrt()
    assert torch.abs(dist1 - dist1_).mean() < 1e-6 and torch.abs(dist2 - dist2_).mean() < 1e-6

    pred_to_gt_normal = F.cosine_similarity(normals_pred, normals_gt[idx1.long()]).mean().item()
    gt_to_pred_normal = F.cosine_similarity(normals_gt, normals_pred[idx2.long()]).mean().item()
    normal_consistency = (pred_to_gt_normal + gt_to_pred_normal) / 2

    # fscore
    threshold = f_thd
    recall = torch.mean((dist1 < threshold).float()).item()
    precision = torch.mean((dist2 < threshold).float()).item()
    fscore = 2 * recall * precision / (recall + precision)
    print(f'cd: {chamfer}, fscore: {fscore}, N.C.: {normal_consistency}')


    metrics = {
        "dist1": pred_to_gt_chamfer,
        "dist2": gt_to_pred_chamfer,
        "chamfer": chamfer,
        "recall": recall,
        "precision": precision,
        "fscore": fscore,
        "normal_dist1": pred_to_gt_normal,
        "normal_dist2": gt_to_pred_normal,
        "normal_consistency": normal_consistency
    }
    return metrics


def record_method_performance3d(uids, mthd_data_root_dict, output_root='outputs/eval3d'):
    mesh_gt_root = mthd_data_root_dict['gt']
    for method in mthd_data_root_dict.keys():
        if method == 'gt': continue

        print(f'processing {method}...')
        mesh_pred_root = mthd_data_root_dict[method]
        metrics_stat = []

        for uid in tqdm(uids):
            mesh_gt_path = os.path.join(mesh_gt_root, uid+'.obj')
            mesh_pred_path = os.path.join(mesh_pred_root, uid, uid+'.obj')      # obj, glb, ply
            assert os.path.isfile(mesh_gt_path) and os.path.isfile(mesh_pred_path)

            try: 
                metrics_of_cur_sample = measure3d(mesh_gt_path, mesh_pred_path, method=method)
                metrics_stat.append(metrics_of_cur_sample)
            except Exception as e:
                print(f'{uid} error!')

        os.makedirs(output_root, exist_ok=True)
        stat_file_path = f'{output_root}/{method}_3dmetrics.json'
        with open(stat_file_path, 'w') as f:
            json.dump(metrics_stat, f, indent=4)
        


def show_stat_res(data_root, methods):
    for method in methods:
        overall_metrics = {
            "chamfer": [],
            "fscore": [],
            "normal_consistency": []
        }

        stat_file_path = f'{data_root}/{method}_3dmetrics.json'
        assert os.path.isfile(stat_file_path)
        with open(stat_file_path, 'r') as f:
            metrics_stat = json.load(f)

        for item_stat in tqdm(metrics_stat):
            cur_sample_chamfer = item_stat['chamfer']
            cur_sample_fscore = item_stat['fscore']
            cur_sample_normal_consistency = item_stat['normal_consistency']

            overall_metrics['chamfer'].append(cur_sample_chamfer)
            overall_metrics['fscore'].append(cur_sample_fscore)
            overall_metrics['normal_consistency'].append(cur_sample_normal_consistency)

        print('method: {}, chamfer: {}, fscore: {}, N.C.: {}'.format(
            method, np.mean(overall_metrics['chamfer']), np.mean(overall_metrics['fscore']), np.mean(overall_metrics['normal_consistency'])
        ))


if __name__ == '__main__':

    stage = 'stage1'
    uids = []                                   # load the uid list of 3D objects for evaluation 
    output_root = 'outputs/eval3d/'             # path to save stat results

    mthd_data_root_dict = {                     # set the predicted_mesh path of each method
            'gt': 'xxx',
            'ours': 'xxx'
    }

    if stage == 'stage1':
        ###############################################################################
        # Stage 1: record 3D metric values of each object from methods
        ###############################################################################
        
        record_method_performance3d(uids, mthd_data_root_dict=mthd_data_root_dict, output_root=output_root)


    elif stage == 'stage2':
        ###############################################################################
        # Stage 2: calculate 3D metric values for each method
        ###############################################################################

        methods = [method for method in mthd_data_root_dict.keys() if method != 'gt']
        show_stat_res(data_root=output_root, methods=methods)