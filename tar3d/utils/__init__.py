import os

import trimesh


def ensure_directory(directory, with_normal=False):
  if not os.path.exists(directory):

    if with_normal:
      os.makedirs(os.path.join(directory, 'rgb'), exist_ok=True)
      os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    else:
      os.makedirs(directory, exist_ok=True)


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