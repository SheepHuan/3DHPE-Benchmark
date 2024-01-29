from .backbone import *
import os.path as osp
import torch
import torch.nn.functional as F
model_urls = {
    'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'ResNext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

BACKBONE_DICT = {
    'LPRES':LpNetResConcat,
    'LPSKI':LpNetSkiConcat,
    'LPWO':LpNetWoConcat
    }

def soft_argmax(heatmaps, joint_num,depth_dim,output_shape ):

    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim*output_shape[0]*output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim, output_shape[0],output_shape[1]))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(1,output_shape[1]+1).to(accu_x.device)
    accu_y = accu_y * torch.arange(1,output_shape[0]+1).to(accu_y.device)
    accu_z = accu_z * torch.arange(1,depth_dim+1).to(accu_z.device)

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out

class CustomNet(nn.Module):
    def __init__(self, cfg):
        super(CustomNet, self).__init__()
        self.cfg = cfg
        self.joint_num = self.cfg.joint_num
        self.backbone = LpNetSkiConcat(input_size=self.cfg.input_shape,
                                        joint_num = self.cfg.joint_num,
                                        embedding_size = self.cfg.embedding_size,
                                        width_mult = self.cfg.width_multiplier)
       

    def forward(self, img, target=None):
        fm = self.backbone(img)
        coord = soft_argmax(fm, self.joint_num, self.cfg.depth_dim, self.cfg.output_shape)

        if self.training:
            target_coord = target['img_3d_pts']
            target_vis = target['joints_vis']

            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2])/3.
            return loss_coord
        else:
            return coord