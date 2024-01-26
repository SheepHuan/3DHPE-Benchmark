import sys
import numpy as np
import os

class Config:
    
    dataset_3d=['Human36M']
    dataset_dir=''
    
    img_shape=[256,256]
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    
    #
    train_batch_size = 64
    test_batch_size = 64
    max_epoch = 25
    lr_dec_epoch = [17, 21]
    lr = 1e-3
    lr_dec_factor = 10
    
    # 
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))