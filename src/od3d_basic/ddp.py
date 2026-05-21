import os
import torch
import torch.distributed as dist

def get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))

def get_rank():
    return int(os.getenv("RANK", 0))

def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))

# Utility: only rank 0 prints logs
def is_main_process(rank):
    return rank == 0

def dist_sync_processes():
    dist.barrier()

def setup_process_group(rank, world_size, master_addr="127.0.0.1", master_port="29500"):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # make sure each process uses its own GPU
    torch.cuda.set_device(rank)

def cleanup_process_group():
    dist.barrier()
    dist.destroy_process_group()