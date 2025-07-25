import numpy as np
import random
import torch


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

