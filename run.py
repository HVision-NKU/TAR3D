import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed


import os
import time
import argparse
# from tar3d.autoregressive.language.t5 import T5Embedder
from tar3d.autoregressive.gpt import GPT_models
from tar3d.autoregressive.generate import generate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tar3d.utils.misc import instantiate_from_config
from tar3d.autoregressive.img_encoder.dino_wrapper import DinoWrapper
from omegaconf import OmegaConf

from PIL import Image
from torchvision.transforms import v2
import numpy as np


def load_tokenizer(config_path=None, ckpt_path=None, device='cuda:0'):
    config = OmegaConf.load(config_path)
    vq_model = instantiate_from_config(config.model.params.module_cfg, device=None, dtype=None)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("sal."):
            new_key = key[4:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    vq_model.load_state_dict(new_state_dict)  
    del checkpoint

    return vq_model


def read_image(path, bg_color = [1., 1., 1.], input_img_size=224):
    pil_img = Image.open(path)

    image = np.asarray(pil_img, dtype=np.float32) / 255.
    alpha = image[:, :, 3:]
    image = image[:, :, :3] * alpha + bg_color * (1 - alpha)

    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

    image = v2.functional.resize(
        image, input_img_size, interpolation=3, antialias=True).clamp(0, 1)
    return image


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------vqmodel-define----------------------------------------------- # 
    vq_model = load_tokenizer(args.tokenizer_config, args.tokenizer_ckpt, device=device)
    print(f"3D tokenizer is loaded")

    # ----------------------------------------------vqmodel-define----------------------------------------------- # 
    img_model = DinoWrapper().to(device) 
    latent_size_tmp = 32
    code_len = 3* (latent_size_tmp ** 2)
    dino_feature_max_len = 197
    max_seq_length = dino_feature_max_len + code_len
    # ----------------------------------------------gpt-init----------------------------------------------- # 
    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=(latent_size ** 2)*3,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        caption_dim=768
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.chkp_path, map_location="cpu")

    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    # del checkpoint
    print(f"gpt model is loaded")
    # ----------------------------------------------gpt-init----------------------------------------------- # 
    
    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 


    # Initialize paths for saving output and the list of input prompts
    output_paths = []
    prompts = ["img_path"]

    output_dir = args.save_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Iterate through all prompts and generate models
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx + 1}/{len(prompts)}: {prompt}")
        # Get the image embedding for the current prompt
        img = read_image(prompt)
        img = img.to(device, non_blocking=True)
        with torch.no_grad():
            caption_embs = img_model(img)
        c_indices = caption_embs.to(device, dtype=precision)
        
        # Start the generation process
        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, (latent_size ** 2) * 3,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
        )
        sampling_time = time.time() - t1
        print(f"Full sampling for prompt {idx + 1} takes about {sampling_time:.2f} seconds.")

        # Decode and save the generated model
        t2 = time.time()
        # output_model_path = os.path.join(output_dir, f'model_{idx + 1}.obj')
        output_model_path = output_paths[idx]
        vq_model.decode_code(
            code_b=index_sample,
            device=index_sample.device,
            batch_size=1,
            bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
            octree_depth=7,
            num_chunks=10000,
            path_to_save_mesh=output_model_path
        )
        decoder_time = time.time() - t2
        print(f"Decoder for prompt {idx + 1} takes about {decoder_time:.2f} seconds.")
        print(f"Model saved to: {output_model_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-config", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['i23d', 't23d'], default="i23d")  
    parser.add_argument("--cls-token-num", type=int, default=197, choices=[120, 197], help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--save-path", type=str, default='')
    parser.add_argument("--chkp-path", type=str, default='')
    args = parser.parse_args()
    main(args)
