# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import time
import argparse
import os

from tar3d.utils.distributed import init_distributed_mode
from tar3d.utils.logger import create_logger
from tar3d.utils.misc import instantiate_from_config

from tar3d.dataset.build import build_dataset
from tar3d.autoregressive.gpt import GPT_models


from datetime import datetime
import inspect
from omegaconf import OmegaConf




#################################################################################
#                             Training Helper Functions                         #
#################################################################################
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


def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(os.path.join(args.results_dir, args.exp_name), exist_ok=True)  # Make results folder (holds all experiment subfolders)
        model_string_name = args.gpt_model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{args.exp_name}/{model_string_name}"

        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")
    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup model
    latent_size = 32
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=(latent_size ** 2) * 3,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        caption_dim=768,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # ----------------------------------------------vqmodel-define----------------------------------------------- #       
    vq_model = load_tokenizer(args.vq_config, args.vq_ckpt, device=device)
    logger.info(f"checkpoint loaded for vq-model")


    # ----------------------------------------------data-define---------------------------------------------- #
    dataset = build_dataset(args, transform=None)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # ----------------------------------------------model-init----------------------------------------------- #
    logger.info(f"checkpoint loaded for gpt")
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
    # ----------------------------------------------model-init----------------------------------------------- #

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y, attn_mask, valid in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            surface = x
            pc = surface[..., 0:3]
            feats = surface[..., 3:]
            with torch.no_grad():
                indices = vq_model.encode_eval(pc, feats)
            x = indices.reshape(pc.shape[0], -1)
            z_indices = x.reshape(x.shape[0], -1)

            c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1])
            assert z_indices.shape[0] == c_indices.shape[0]
            attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]) # (bs, n_head, seq_len, seq_len)
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, mask=attn_mask[:, :, :-1,:-1], valid=valid)
            
            print('[{}] step: {}, loss: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_steps, loss))
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    # if not args.no_local_save:

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        if epoch >= 40:
            if rank == 0:
                if not args.no_compile:
                    model_weight = model.module._orig_mod.state_dict()
                else:
                    model_weight = model.module.state_dict()  
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "epoch": epoch,
                    "args": args
                }
                # if not args.no_local_save:
                checkpoint_path = f"{checkpoint_dir}/last.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            dist.barrier()
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='i23d')
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--vq-config", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')

    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['i23d', 't23d'], default="i23d")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=197, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results_vqi23d")
    parser.add_argument("--dataset", type=str, default='i23d')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=80)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    args = parser.parse_args()
    
    main(args)
