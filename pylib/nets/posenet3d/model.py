import torch
import torch.nn as nn
from torch.nn import functional as F
from .resnet import ResNetBackbone
import torch
import torch.nn as nn
from torch.nn import functional as F
# from nets.resnet import ResNetBackbone
# from config import cfg

class RootNet(nn.Module):

    def __init__(self,output_shape):
        self.inplanes = 2048
        self.outplanes = 256
        self.output_shape = output_shape
        super(RootNet, self).__init__()
        self.deconv_layers = self._make_deconv_layer(3)
        self.xy_layer = nn.Conv2d(
            in_channels=self.outplanes,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=1, 
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        # x,y
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)
        xy = xy.view(-1,1,self.output_shape[0]*self.output_shape[1])
        xy = F.softmax(xy,2)
        xy = xy.view(-1,1,self.output_shape[0],self.output_shape[1])

        hm_x = xy.sum(dim=(2))
        hm_y = xy.sum(dim=(3))

        coord_x = hm_x * torch.arange(self.output_shape[1]).to(hm_x.device).float()
        coord_y = hm_y * torch.arange(self.output_shape[0]).to(hm_x.device).float()
        
        coord_x = coord_x.sum(dim=2)
        coord_y = coord_y.sum(dim=2)

        # z
        img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2) # global average pooling
        img_feat = torch.unsqueeze(img_feat,2)
        img_feat = torch.unsqueeze(img_feat,3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1,1)
        depth = gamma * k_value.view(-1,1)

        coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return coord

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.xy_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class ResRootNet(nn.Module):
    def __init__(self, backbone, root):
        super(ResRootNet, self).__init__()
        self.backbone = backbone
        self.root = root

    def forward(self, input_img, k_value=1):
        fm = self.backbone(input_img)
        coord = self.root(fm, k_value)
        return coord



class HeadNet(nn.Module):

    def __init__(self, 
                 joint_num,
                 depth_dim,
                 output_shape):
        self.inplanes = 2048
        self.outplanes = 256
        
        self.joint_num = joint_num
        self.depth_dim = depth_dim
        self.output_shape = output_shape

        super(HeadNet, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=joint_num * depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)



class ResPoseNet(nn.Module):
    def __init__(self, backbone, head, joint_num,depth_dim,output_shape):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.depth_dim = depth_dim
        self.joint_num = joint_num
        self.output_shape = output_shape

    def forward(self, input_img):
        fm = self.backbone(input_img)
        hm = self.head(fm)
        coord = self.soft_argmax(hm)
        return coord
       

    def soft_argmax(self,heatmaps):

        heatmaps = heatmaps.reshape((-1, self.joint_num, self.depth_dim*self.output_shape[0]*self.output_shape[1]))
        heatmaps = F.softmax(heatmaps, 2)
        heatmaps = heatmaps.reshape((-1, self.joint_num, self.depth_dim, self.output_shape[0], self.output_shape[1]))

        accu_x = heatmaps.sum(dim=(2,3))
        accu_y = heatmaps.sum(dim=(2,4))
        accu_z = heatmaps.sum(dim=(3,4))

        accu_x = accu_x * torch.arange(self.output_shape[1]).float().cuda()[None,None,:]
        accu_y = accu_y * torch.arange(self.output_shape[0]).float().cuda()[None,None,:]
        accu_z = accu_z * torch.arange(self.depth_dim).float().cuda()[None,None,:]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

        return coord_out


class RootNet3D(nn.Module):
    def __init__(self, cfg) -> None:
        super(RootNet3D,self).__init__()
        self.cfg = cfg
        self.rootnet = ResRootNet(ResNetBackbone(self.cfg.resnet_type),RootNet(cfg.output_shape))
    
    def forward(self,img,key_value,target=None):
        coord = self.rootnet(img,key_value)
        if self.training:
            # 这里通过img_3d_pts计算深度loss,x,y,z都经过归一化了
            target_coord = target['img_3d_pts']
            target_vis = target['joints_vis']
          
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2])/3.0
            return loss_coord
        else:
            return coord

class PoseNet3D(nn.Module):
    
    def __init__(self, cfg) -> None:
        super(PoseNet3D,self).__init__()
        self.cfg = cfg
    
        self.posenet = ResPoseNet(ResNetBackbone(cfg.resnet_type),HeadNet(cfg.joint_num,cfg.depth_dim,cfg.output_shape),cfg.joint_num,cfg.depth_dim,cfg.output_shape)

    def forward(self,img,target=None):
        pred_img_relative_coords = self.posenet(img)
        
        if self.training:
            # 这里通过img_3d_pts计算深度loss,x,y,z都经过归一化了
            target_coord = target['img_3d_pts']
            target_vis = target['joints_vis']
          
            loss_coord = torch.abs(pred_img_relative_coords - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2])/3.0
            return loss_coord
        else:
            return pred_img_relative_coords
