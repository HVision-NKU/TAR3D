import os
import json
import numpy as np
import trimesh
from tqdm import tqdm

import torch
import torchvision

import lpips
import math
from skimage.metrics import structural_similarity as compare_ssim
import open_clip
from PIL import Image

from tar3d.utils import ensure_directory, load_mesh
from tar3d.evaluation.myrender.render import render_mesh


def transform_mesh(mesh_pred, method='gt'):
    # transform to align the view of meshes from different method; judge according to the render images.
    if method in ['method1']:           # take a example
        z_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1])  # 旋转矩阵：绕Z轴左转90度
        x_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])  # 旋转矩阵：绕X轴左转90度
        y_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0]) # 旋转矩阵：绕Y轴左转180度
        mesh_pred.apply_transform(z_rotation_matrix)
        mesh_pred.apply_transform(x_rotation_matrix)
        mesh_pred.apply_transform(y_rotation_matrix)
    elif method == 'gt':                # take a example
        x_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        y_rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(32), [0, 1, 0])
        mesh_pred.apply_transform(x_rotation_matrix)
        mesh_pred.apply_transform(y_rotation_matrix)
    return mesh_pred


def load_mesh(mesh_path):
    mesh_pred = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh_pred, trimesh.Scene):
        meshes = mesh_pred.dump()
        mesh_pred = trimesh.util.concatenate(meshes)
    center = mesh_pred.bounding_box.centroid
    mesh_pred.apply_translation(-center)
    scale = max(mesh_pred.bounding_box.extents)
    mesh_pred.apply_scale(2 / scale)
    return mesh_pred

def generate_images_for_mesh(mesh_path, output_root, uid=0, view_idx=-1, method_name='gt', render_resolution=299):
    '''
    method_name 方法
    uid 表示mesh索引
    view_idx 表示视角索引
    '''
    mesh_pred = load_mesh(mesh_path)
    mesh = transform_mesh(mesh_pred, method_name)

    if view_idx == -1:
        for view_idx in tqdm(range(20)):
            image, normal = render_mesh(mesh, index=view_idx, resolution=render_resolution)

            ensure_directory(f"{output_root}/view_{view_idx}", with_normal=True)
            torchvision.utils.save_image(torch.from_numpy(image.copy()/255).permute(2, 0, 1), 
                                         f"{output_root}/view_{view_idx}/rgb/{uid}_{method_name}.png")
            torchvision.utils.save_image(torch.from_numpy(normal.copy()/255).permute(2, 0, 1), 
                                         f"{output_root}/view_{view_idx}/normal/{uid}_{method_name}.png")
    else:
        image, normal = render_mesh(mesh, index=view_idx, resolution=render_resolution)

        ensure_directory(f"{output_root}/view_{view_idx}", with_normal=True)
        torchvision.utils.save_image(torch.from_numpy(image.copy()/255).permute(2, 0, 1), 
                                     f"{output_root}/view_{view_idx}/rgb/{uid}_{method_name}.png")
        torchvision.utils.save_image(torch.from_numpy(normal.copy()/255).permute(2, 0, 1), 
                                     f"{output_root}/view_{view_idx}/normal/{uid}_{method_name}.png")


def test_mesh_render():
    uid = 'jacket'
    mesh_pred_path = f'assets/examples/{uid}.obj'
    output_root = 'outputs/tmp_eval2d'
    view_idx = -1
    method = 'gt'
    render_resolution = 224
    generate_images_for_mesh(mesh_pred_path, 
                            output_root, 
                            uid=uid, view_idx=view_idx,         # which object, which view
                            method_name=method,                 # which method
                            render_resolution=render_resolution)


def render_mthd_meshes(uids, mthd_data_root_dict, output_root='outputs/eval2d/', render_resolution=224, view_idx=-1):
    for uid in tqdm(sorted(uids)):
        for method, mesh_pred_root in mthd_data_root_dict.items():
            if method == 'ours':
                mesh_pred_path = os.path.join(mesh_pred_root, uid, uid+'.obj')      # obj, glb, ply
            else:
                mesh_pred_path = os.path.join(mesh_pred_root, uid+'.obj')           
            assert os.path.isfile(mesh_pred_path)

            generate_images_for_mesh(mesh_pred_path, 
                                     output_root, 
                                     i=uid, j=view_idx,         # which object, which view
                                     method_name=method,        # which method
                                     render_resolution=render_resolution)

###############################################################################

def load_im(path, color=[1., 1., 1.]):
    pil_img = Image.open(path)

    image = np.asarray(pil_img, dtype=np.float32) / 255.

    if image.shape[-1] == 4:
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)
    return image

# PSNR
def get_psnr(gt_img_path=None, pred_img_path=None):
    gt_img = load_im(gt_img_path)
    pred_img = load_im(pred_img_path)
    score = compt_psnr(gt_img, pred_img)

    return score

def compt_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# SSIM
def get_ssim(gt_img_path=None, pred_img_path=None):
    gt_img = load_im(gt_img_path)
    pred_img = load_im(pred_img_path)

    # gt_img = gt_img.astype(np.uint8)
    # pred_img = pred_img.astype(np.uint8)
    score = compare_ssim(gt_img, pred_img, data_range=gt_img.max() - gt_img.min(), gaussian_weights=True, sigma=1.5, channel_axis=-1, use_sample_covariance=False, full=True)

    return score[0]


# CLIP
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


model_name = ('ViT-B-32', 'laion400m_e31')
clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1],device=device)
tokenizer = open_clip.get_tokenizer(model_name[0])

def get_clip(gt_img_path=None, pred_img_path=None):
    gt_img = preprocess(Image.open(gt_img_path))
    pred_img = preprocess(Image.open(pred_img_path))

    gt_img = gt_img.unsqueeze(0).to(device=device)
    pred_img = pred_img.unsqueeze(0).to(device=device)

    gt_img = clip_model.encode_image(gt_img)
    pred_img = clip_model.encode_image(pred_img)

    gt_img /= gt_img.norm(dim=-1, keepdim=True)
    pred_img /= pred_img.norm(dim=-1, keepdim=True)

    clip_score = 100 * pred_img @ gt_img.T

    return clip_score.mean().item()


# LIPIS
cal_lpips = lpips.LPIPS(net='alex')
def get_lpips(gt_img_path=None, pred_img_path=None):

    gt_img = load_im(gt_img_path)
    pred_img = load_im(pred_img_path)

    # print(type(gt_img), type(pred_img))
    gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).contiguous().float().unsqueeze(0)
    pred_img = torch.from_numpy(pred_img).permute(2, 0, 1).contiguous().float().unsqueeze(0)

    score = cal_lpips(gt_img, pred_img).item()
    # print(score)
    return score

def eval_metrics(uids, data_root, method='ours', data_type='rgb'):
    print(f'processing {method}....')
    view_folders = sorted([view_folder for view_folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, view_folder))])

    all_metrics ={'psnr': [], 'ssim': [], 'clip': [], 'lpips': []}
    for uid in tqdm(uids):
        cur_metrics = {'psnr': [], 'ssim': [], 'clip': [], 'lpips': []}
        for view in view_folders:
            gt_file_path = os.path.join(data_root, view, data_type, f'gt_{uid}.png')
            if not os.path.isfile(gt_file_path): continue

            pred_file_path = os.path.join(data_root, view, data_type, f'{uid}_{method}.png')
            if not os.path.isfile(pred_file_path): continue

            cur_metrics['psnr'].append(
                get_psnr(gt_img_path=gt_file_path, pred_img_path=pred_file_path)
            )
            cur_metrics['ssim'].append(
                get_ssim(gt_img_path=gt_file_path, pred_img_path=pred_file_path)
            )

            cur_metrics['lpips'].append(
                get_lpips(gt_img_path=gt_file_path, pred_img_path=pred_file_path)
            )
            cur_metrics['clip'].append(
                get_clip(gt_img_path=gt_file_path, pred_img_path=pred_file_path)
            )
 
        uid_psnr = sum(cur_metrics['psnr']) / len(cur_metrics['psnr'])
        uid_ssim = sum(cur_metrics['ssim']) / len(cur_metrics['ssim'])
        uid_clip = sum(cur_metrics['clip']) / len(cur_metrics['clip'])
        uid_lpips = sum(cur_metrics['lpips']) / len(cur_metrics['lpips'])

        # print(uid_psnr, uid_ssim, uid_clip, uid_lpips)
        all_metrics['psnr'].append(uid_psnr)
        all_metrics['ssim'].append(uid_ssim)
        all_metrics['clip'].append(uid_clip)
        all_metrics['lpips'].append(uid_lpips)

    with open(f'{data_root}/{method}_{data_type}_2dmetrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    

    psnr_value = sum(all_metrics['psnr']) / len(all_metrics['psnr'])
    ssim_value = sum(all_metrics['ssim']) / len(all_metrics['ssim'])
    clip_value = sum(all_metrics['clip']) / len(all_metrics['clip'])
    lpips_value = sum(all_metrics['lpips']) / len(all_metrics['lpips'])

    print(f'method: {method}, psnr: {psnr_value}, ssim: {ssim_value}, clip: {clip_value}, lpips: {lpips_value}')


if __name__ == "__main__":

    stage = 'stage1'
    uids = []                                           # load the uid list of 3D objects for evaluation 
    output_root = 'outputs/eval2d_renders/'             # path to save renders

    if stage == 'stage1':
        ###############################################################################
        # Stage 1: render mesh for 2D images
        ###############################################################################
        
        mthd_data_root_dict = {                         # set the predicted_mesh path of each method
            'gt': 'xxx',
            'ours': 'xxx'
        }
        
        render_resolution = 224                     
        view_idx = -1                                   # -1: all 20 views, or specific view: int
        render_mthd_meshes(uids, mthd_data_root_dict, output_root=output_root, render_resolution=render_resolution, view_idx=-1)

    elif stage == 'stage2':
        ###############################################################################
        # Stage 2: calculate 2D metric values
        ###############################################################################
        mthd_names = ['ours', 'xxx']

        for mthd in mthd_names:
            eval_metrics(uids=uids, data_root=output_root, method=mthd, data_type='rgb')
            # eval_metrics(uids=uids, data_root=output_root, method=mthd, data_type='normal')
        