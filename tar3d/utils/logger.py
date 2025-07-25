import logging
import torch.distributed as dist
import os

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    log_count = len([filename for filename in os.listdir(logging_dir) if filename.endswith('.txt') and filename.startswith('log')])
    if log_count == 0:
        log_name = 'log'
    else:
        log_name = f'log{log_count:02d}'

    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{log_name}.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger