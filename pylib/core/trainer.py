import torch
import os.path as osp
import glob
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast as autocast
from .logger import colorlogger
import torch.distributed as dist
import time
from torch.nn import SyncBatchNorm
import numpy as np
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 10:
            self.warm_up += 1
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff


class Base:
    start_epoch = 0
    
    def __init__(self,cfg) -> None:
        self.cfg = cfg

        print(self.cfg.local_rank)
        
        if self.cfg.distributed:
            if self.cfg.local_rank == 0:
                self.logger = colorlogger(self.cfg.checkpoints_save_dir)
            self.device = torch.device(self.cfg.local_rank)
        else:
            self.logger = colorlogger(self.cfg.checkpoints_save_dir)
            self.device = torch.device(f"cuda:{self.cfg.local_rank}") 
    
    def save_model(self,epoch,model,optimizer):
        if self.cfg.local_rank!=0 and self.cfg.distributed:
            pass
        file_path = osp.join(self.cfg.checkpoints_save_dir,f'snapshot_{epoch}.pth.tar')
        torch.save({
            "network": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, file_path)
        self.logger.info(f"Save model at epoch {epoch}")
    
    
    def load_model(self,model,optimizer):
        start_epoch=0
        if self.cfg.checkpoints_pretrained_dir!=None:
            model_file_list = glob.glob(osp.join(self.cfg.checkpoints_pretrained_dir,'*.pth.tar'))
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
            ckpt = torch.load(osp.join(self.cfg.checkpoints_pretrained_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar'))
            start_epoch = ckpt['epoch'] + 1
            
            if self.cfg.distributed==False:
                ckpt['network'] = {k.replace('module.', ''): v for k, v in ckpt['network'].items()}
            
            model.load_state_dict(ckpt['network'])
            optimizer.load_state_dict(ckpt['optimizer'])
        elif self.cfg.distributed:
            
            # 主进程保存权重,其他进程加载主进程保存的权重
            checkpoint_path = osp.join("/tmp",'tmp.model.pth')
            if self.cfg.local_rank==0:
                torch.save({
                    "network": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, checkpoint_path)
            dist.barrier()
            # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
            ckpt=torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(ckpt["network"])
            optimizer.load_state_dict(ckpt["optimizer"])
            
        return start_epoch,model,optimizer
        
        

class Trainer(Base):
    # 混合精度训练
    # 分布式训练
    
    
    def genertate_dataset(self):
        from datasets import PoseNet3dDataset
        from torch.utils.data import DataLoader
        self.train_dataset = PoseNet3dDataset(self.cfg,is_train=True)
        
        if self.cfg.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            train_batch_sampler=torch.utils.data.BatchSampler(train_sampler,self.cfg.train_batch_size,drop_last=True)
            self.train_dataloader = DataLoader(self.train_dataset, 
                                            pin_memory=True,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=self.cfg.num_workers,
                                            )
        else:
           self.train_dataloader = DataLoader(self.train_dataset, 
                                            pin_memory=True,
                                            batch_size=self.cfg.train_batch_size,
                                            shuffle=True,
                                            num_workers=self.cfg.num_workers,
                                            ) 
        
        self.test_dataset = PoseNet3dDataset(self.cfg,is_train=False)
        
        if self.cfg.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)
            self.test_dataloader = DataLoader(self.test_dataset,
                                            batch_size=self.cfg.test_batch_size, 
                                            pin_memory=True,
                                            sampler=test_sampler,
                                        )
        else:
            self.test_dataloader = DataLoader(self.test_dataset,
                                            batch_size=self.cfg.test_batch_size, 
                                            pin_memory=True,
                                            shuffle=False
                                        )
    
    def __init__(self,cfg) -> None:
        super(Trainer,self).__init__(cfg)
        
        self.setup_seed(2024)
        self.genertate_dataset()
        from nets import PoseNet3D,CustomNet
        
        
        
        # 根据配置文件初始化这些参数
        # self.model = PoseNet3D(cfg)
        self.model = CustomNet(cfg)
        # 定义数据集
        
            
        self.model = self.model.to(self.device)
        if self.cfg.distributed:
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        # 加载权重
        self.start_epoch,self.model,self.optimizer = self.load_model(self.model,self.optimizer)
        
        

        self.train_timer = Timer()
    
    
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    
    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr


    def set_lr(self, epoch):
        for e in self.cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.cfg.lr_dec_epoch[-1]:
            idx = self.cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_epoch))
                
                
    def train(self,loss_func,evalute_func=None,visual_func=None):
        cudnn.fastest = True
        cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler()
        
        self.model = self.model.to(self.device)
        iter_per_epoch = len(self.train_dataloader)
        
        # self.test(evalute_func)
        
        for epoch in range(self.start_epoch, self.cfg.end_epoch):
            self.model.train()
            self.set_lr(epoch)
            for itr, data in enumerate(self.train_dataloader):
                self.train_timer.tic()
                
                self.optimizer.zero_grad()
                
                with autocast(dtype=torch.float16):
                
                    loss = loss_func(self.model,data,self.device)
                # 反向传播在autocast上下文之外
                # Scales loss. 为了梯度放大.
                scaler.scale(loss).backward()
                # scaler.step() 首先把梯度的值unscale回来.
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(self.optimizer)

                # 准备着，看是否要增大scaler
                scaler.update()
                self.train_timer.toc()
                
                if (self.cfg.distributed and self.cfg.local_rank==0) or (self.cfg.distributed==False):
                    screen = [
                    'Epoch %d/%d, itr %d/%d' % (epoch, self.cfg.end_epoch, itr, iter_per_epoch),
                    'loss: %.4f' % (loss.detach()),
                    "speed: %.2f s/iter, %.2f h/epoch" % (self.train_timer.average_time,self.train_timer.average_time/3600.0*iter_per_epoch)
                    ]
                    self.logger.info(" ".join(screen))
                
            
            
            # if epoch % 5 == 0 and epoch != 0:
            #     metrics = self.test(evalute_func)
                
            
            if (self.cfg.distributed and self.cfg.local_rank==0) or (self.cfg.distributed==False):      
                self.save_model(epoch,self.model,self.optimizer)
    
    def test(self,evalute_func):
        metrics = {}
        self.model.eval()
        iter_per_epoch = len(self.test_dataloader)
        test_timer = Timer()
        for itr, data in enumerate(self.test_dataloader):
            
            test_timer.tic()
            # with autocast(dtype=torch.float16):   
            metrics = evalute_func(itr,metrics,self.model,data,self.device)
            
            test_timer.toc()
            if (self.cfg.distributed and self.cfg.local_rank==0) or (self.cfg.distributed==False):
                screen = [
                        'Eval Epoch itr %d/%d' % (itr, iter_per_epoch),
                        "speed: %.2f s/iter, %.2f h/epoch" % (test_timer.average_time,test_timer.average_time/3600.0*iter_per_epoch)
                        ]
                self.logger.info(" ".join(screen))
            
               
            
        for key,value in metrics.items():
            metrics[key]=np.mean(np.asanyarray(value))
        
        if (self.cfg.distributed and self.cfg.local_rank==0) or (self.cfg.distributed==False):
            screen = [ f"Eval Metric",] + [
                "%s: %0.4f" % (key,value) for key,value in metrics.items()
            ]
            self.logger.info(" ".join(screen))
          
    

    
       
                
    