import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from functools import partial
import einops
import trimesh

import torch
from tar3d.utils.misc import instantiate_from_config
from tar3d.tokenizer.tsal.inference_utils import extract_geometry


def load_surface(pc_path, pc_size=81920):
    with np.load(pc_path) as input_pc:
        surface = input_pc['points']
        normal = input_pc['normals']
    
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], pc_size, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0)
    
    return surface


def point2mesh(
    pc, feats, model, 
    path_to_save_mesh=None, 
    bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25), 
    octree_depth=7, num_chunks=10000, post_processing=False
):
    
    latents, center_pos, posterior = model.sal.encode(pc, feats)
    # latents = model.sal.decode(latents)  # latents: [bs, num_latents, dim]
    B = latents.shape[0]
    latent_z = latents.view(B, 3*model.sal.triplane_res, model.sal.triplane_res, 768).permute(0, 3, 1, 2).contiguous()

    latent_z = model.sal.quant_conv(latent_z)
    latent_z = einops.rearrange(latent_z, 'B C (P H) W -> (B P) C H W', P=3)
    latent_z, _, _, _  = model.sal.quant(latent_z)
    latent_z = einops.rearrange(latent_z, '(B P) C H W -> B C (P H) W', B=B)
    latent_z = model.sal.post_quant_conv(latent_z)
    latents = model.sal.decode_latent_to_triplane(latent_z)
    
    geometric_func = partial(model.sal.decode_triplane_to_sdf, planes=latents)
    
    mesh_v_f, has_surface = extract_geometry(
        geometric_func=geometric_func,
        device=pc.device,
        batch_size=pc.shape[0],
        bounds=bounds,
        octree_depth=octree_depth,
        num_chunks=num_chunks,
    )
    recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])

    if post_processing:
        components = recon_mesh.split(only_watertight=False)
        components = [c for c in components if c.area > 0.05]
        recon_mesh = trimesh.util.concatenate(components)
    
    recon_mesh.export(path_to_save_mesh)    

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {path_to_save_mesh}')
    print(f'-----------------------------------------------------------------------------')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", type=str, default="configs/vqvae3d.yaml")
    parser.add_argument("--input_file", type=str, default='assets/surface.npz')
    parser.add_argument("--pc_size", type=int, default=81920)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='outputs/vqvae')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.base)

    model = instantiate_from_config(config.model, ckpt_path=args.ckpt_path)
    model = model.cuda()
    model = model.eval()


    surface = load_surface(args.input_file, args.pc_size)
    pc = surface[..., 0:3].cuda()
    feats = surface[..., 3:].cuda()


    uid = os.path.splitext(os.path.basename(args.input_file))[0]
    path_to_save_mesh = os.path.join(args.output_dir, f'{uid}.obj')
    with torch.no_grad():
        point2mesh(pc=pc, feats=feats, model=model, path_to_save_mesh=path_to_save_mesh)

