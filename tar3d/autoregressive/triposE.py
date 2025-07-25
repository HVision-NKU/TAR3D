from einops import rearrange
import torch
import numpy as np
from tqdm import tqdm



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (seq_len, head_dim // 2, 2)
    return cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    return cache

def repeat_pos2d_for_triplane(x):
    l = x.shape[0]
    x = x.repeat(3, 1, 1)

    x1, x2, x3 = torch.split(x, [l, l, l], dim=0)

    pos_embed_list = []
    for idx in tqdm(range(l)):
        pos_embed_list.append(x1[idx].unsqueeze(0))
        pos_embed_list.append(x2[idx].unsqueeze(0))
        pos_embed_list.append(x3[idx].unsqueeze(0))

    x = torch.cat(pos_embed_list, dim=0)
    return x


def precompute_freqs_cis_3d(block_size=3072, n_elem=64, rope_base=10000, cls_token_num=120):
    
    grid_size = int((block_size//3) ** 0.5)
    assert grid_size * grid_size * 3 == block_size
    freqs_cis_2d = precompute_freqs_cis_2d(grid_size, n_elem, rope_base)

    # print('2d, before: ', freqs_cis_2d.shape)
    freqs_cis_2d = repeat_pos2d_for_triplane(freqs_cis_2d)
    # print('2d, after: ', freqs_cis_2d.shape)

    freqs_cis_1d = precompute_freqs_cis(3, n_elem, rope_base)
    # print('1d, before: ', freqs_cis_1d.shape)
    freqs_cis_1d = freqs_cis_1d.repeat(block_size//3, 1, 1)
    # print('1d, after: ', freqs_cis_1d.shape)

    cache = freqs_cis_2d + freqs_cis_1d


    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) 
    # print('final: ', cond_cache.shape)

    return cond_cache

# pos_embed = precompute_freqs_cis_3d(block_size=3072)
# print(pos_embed.shape)