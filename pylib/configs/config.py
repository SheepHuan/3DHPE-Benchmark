import sys
import numpy as np
import os
import torch
from datetime import datetime
import torch.distributed as dist
# 获取当前的时间日期
now = datetime.now()
# 格式化时间日期字符串
formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")

class Config:
    
    dataset_3d=['Human36M']
    dataset_dir='/dataset/Human3.6M'
    
    
    rgb_pixel_mean = (0.485, 0.456, 0.406)
    rgb_pixel_std = (0.229, 0.224, 0.225)
    
    #
    train_batch_size = 64
    test_batch_size = 48
    end_epoch = 20
    lr_dec_epoch = [6, 8, 10, 16]
    lr = 1e-3
    lr_dec_factor = 10
    
    # 
    num_workers = 56
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    checkpoints_pretrained_dir = None
    # checkpoints_pretrained_dir = "/root/code/3DHPE-Benchmark/checkpoints/2024_01_28_14_15_27"
    
    distributed = True
    local_rank = 0
    
    def __init__(self) -> None:
        if self.distributed==True:
            self.init_distributed_mode()
        
        self.init_dir()
   
    
    def init_dir(self):
        if self.distributed==True:
            if self.local_rank==0:
                self.checkpoints_save_dir = f"checkpoints/{formatted_datetime}"
                os.makedirs(self.checkpoints_save_dir,exist_ok=True)
        else:
            self.checkpoints_save_dir = f"checkpoints/{formatted_datetime}"
            os.makedirs(self.checkpoints_save_dir,exist_ok=True)      
    
    
    def init_distributed_mode(self):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
        if'RANK'in os.environ and'WORLD_SIZE'in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            # LOCAL_RANK代表某个机器上第几块GPU
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            self.lr *= self.world_size # 学习率要根据并行GPU的数倍增
        elif'SLURM_PROCID'in os.environ:
            self.rank = int(os.environ['SLURM_PROCID'])
            self.local_rank = self.rank % torch.cuda.device_count()
        

        
        self.dist_backend = 'nccl'# 通信后端，nvidia GPU推荐使用NCCL
        dist.init_process_group(backend=self.dist_backend)
        torch.cuda.set_device(self.local_rank)  # 对当前进程指定使用的GPU
        
        dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续