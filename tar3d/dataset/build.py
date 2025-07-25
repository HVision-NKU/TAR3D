from .gpt import build_ulipiTo3D, build_uliptTo3D


def build_dataset(args, **kwargs):
    if args.dataset == 't23d':
        return build_uliptTo3D(args, **kwargs)
    if args.dataset == 'i23d':
        return build_ulipiTo3D(args, **kwargs)

    raise ValueError(f'dataset {args.dataset} is not supported')


