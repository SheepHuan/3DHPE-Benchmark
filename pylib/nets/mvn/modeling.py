import torch
from torch import nn
from .models.triangulation import VolumetricTriangulationNet

class MVNet(nn.Module):
    
    def __init__(self,cfg):
        self.cfg = cfg
        
        # 需要poseresnet计算heatmap
        
        
        
    def forward(self):
        pass