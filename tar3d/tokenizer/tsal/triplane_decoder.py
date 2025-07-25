import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import Optional


class ResNetBlock(nn.Module):
    def __init__(self, device: Optional[torch.device], dtype: Optional[torch.dtype], in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype),
                nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class UpsampleResNet_ConvTranspose(nn.Module):
    def __init__(self, device: Optional[torch.device], dtype: Optional[torch.dtype], in_channels=16, hidden_channels=32):
        super(UpsampleResNet_ConvTranspose, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                            kernel_size=2, stride=2,
                                            padding=0, output_padding=0, device=device, dtype=dtype)
        self.upsample2 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                            kernel_size=2, stride=2,
                                            padding=0, output_padding=0, device=device, dtype=dtype)
        self.upsample3 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                            kernel_size=2, stride=2,
                                            padding=0, output_padding=0, device=device, dtype=dtype)

        self.block1 = ResNetBlock(in_channels=in_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block2 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block3 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block4 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block5 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)

    def forward(self, x):
        x = self.block1(x)
        x = self.upsample1(x)  # 32x32 -> 64x64
        x = self.block2(x)    # 64x64 -> 64x64
        x = self.block3(x)    # 64x64 -> 64x64
        x = self.upsample2(x)  # 64x64 -> 128x128
        x = self.block4(x)    # 128x128 -> 256x256
        x = self.upsample3(x)  # 128x128 -> 256x256
        x = self.block5(x)
        return x

class UpsampleResNet(nn.Module):
    def __init__(self, device: Optional[torch.device], dtype: Optional[torch.dtype], in_channels=16, hidden_channels=32):
        super(UpsampleResNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.block1 = ResNetBlock(in_channels=in_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block2 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block3 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block4 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block5 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)

    def forward(self, x):
        x = self.block1(x)
        x = self.upsample(x)  # 32x32 -> 64x64
        x = self.block2(x)    # 64x64 -> 64x64
        x = self.block3(x)    # 64x64 -> 64x64
        x = self.upsample(x)  # 64x64 -> 128x128
        x = self.block4(x)    # 128x128 -> 256x256
        x = self.upsample(x)  # 128x128 -> 256x256
        x = self.block5(x)
        return x
    
class UpsampleResNet_4layer(nn.Module):
    def __init__(self, device: Optional[torch.device], dtype: Optional[torch.dtype], in_channels=16, hidden_channels=32):
        super(UpsampleResNet_4layer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.block1 = ResNetBlock(in_channels=in_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block2 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block3 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block4 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)
        self.block5 = ResNetBlock(in_channels=hidden_channels, out_channels=hidden_channels, device=device, dtype=dtype)

    def forward(self, x):
        # x = self.block1(x)
        # x = self.upsample(x)  # 32x32 -> 64x64
        x = self.block2(x)    # 64x64 -> 64x64
        x = self.block3(x)    # 64x64 -> 64x64
        x = self.upsample(x)  # 64x64 -> 128x128
        x = self.block4(x)    # 128x128 -> 256x256
        x = self.upsample(x)  # 128x128 -> 256x256
        x = self.block5(x)
        return x

class Pre_upsampleResNet(nn.Module):
    def __init__(self, device: Optional[torch.device], dtype: Optional[torch.dtype], in_channels=16, hidden_channels=32):
        super(Pre_upsampleResNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.block = ResNetBlock(in_channels=in_channels, out_channels=hidden_channels, device=device, dtype=dtype)

    def forward(self, x):

        x = self.block(x)
        x = self.upsample(x)  # 32x32 -> 64x64

        return x
    
# ****************************************************************************************************


class OSGDecoder(nn.Module):
    """
    Triplane decoder that gives RGB and sigma values from sampled features.
    Using ReLU here instead of Softplus in the original implementation.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
    """
    def __init__(self, device: Optional[torch.device], dtype: Optional[torch.dtype], n_features: int,
                 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU, is_sdf_logits: bool = True):
        super().__init__()

        self.is_sdf_logits = is_sdf_logits

        self.net_sdf = nn.Sequential(
            nn.Linear(3 * n_features, hidden_dim, device=device, dtype=dtype),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1, device=device, dtype=dtype),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def post_process(self, sdf_logits, s=1/512):
        output = torch.full_like(sdf_logits, 0.5)           # 初始化为0.5
        output[sdf_logits < -s] = 1                         # 小于 -s 的值设为 1
        output[sdf_logits > s] = 0                          # 大于 s 的值设为 0
        mask = (sdf_logits >= -s) & (sdf_logits <= s)       # 介于 -s 和 s 之间的值
        output[mask] = 0.5 - 0.5 * sdf_logits[mask] / s     # 输出0.5 - 0.5 * logits / s
        return output        

    def get_geometry_prediction(self, sampled_features):
        _N, n_planes, _M, _C = sampled_features.shape
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
        sdf = self.net_sdf(sampled_features)

        if self.is_sdf_logits: 
            return sdf
        else:
            sdf = self.post_process(sdf)
            return sdf
    

class TriplaneSynthesizer(nn.Module):
    """
    Synthesizer that renders a triplane volume with planes and a camera.
    
    Reference:
    EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L19
    """

    DEFAULT_RENDERING_KWARGS = {
        'ray_start': 'auto',
        'ray_end': 'auto',
        'box_warp': 2.,
        'white_back': True,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'sampler_bbox_min': -1.,
        'sampler_bbox_max': 1.,
    }

    def __init__(self, triplane_dim: int, samples_per_ray: int, device: Optional[torch.device], dtype: Optional[torch.dtype]):
        super().__init__()

        # attributes
        self.triplane_dim = triplane_dim
        self.rendering_kwargs = {
            **self.DEFAULT_RENDERING_KWARGS,
            'depth_resolution': samples_per_ray // 2,
            'depth_resolution_importance': samples_per_ray // 2,
        }

        # modules
        self.plane_axes = generate_planes()
        self.decoder = OSGDecoder(n_features=triplane_dim, device=device, dtype=dtype)

    def get_geometry_prediction(self, planes, sample_coordinates):
        plane_axes = self.plane_axes.to(planes.device)
        sampled_features = sample_from_planes(
            plane_axes, planes, sample_coordinates, padding_mode='zeros', box_warp=self.rendering_kwargs['box_warp'])

        sdf = self.decoder.get_geometry_prediction(sampled_features)
        return sdf



# ****************************************************************************************************



def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.

    Bugfix reference: https://github.com/NVlabs/eg3d/issues/67
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)
    dtype = plane_features.dtype

    coordinates = (2/box_warp) * coordinates # add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(
        plane_features, 
        projected_coordinates.to(dtype), 
        mode=mode, 
        padding_mode=padding_mode, 
        align_corners=False,
    ).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)


    bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out