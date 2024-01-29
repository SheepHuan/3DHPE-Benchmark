from .single_h36m import SingleHuman36M
import numpy as np
import torchvision.transforms as transforms
import torch
import math
from .single_h36m import trans_point2d,gen_trans_from_patch_cv

class PoseNet3dDataset(SingleHuman36M):
    
    def __init__(self,cfg,is_train=False):
        super(PoseNet3dDataset,self).__init__(cfg,is_train)
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=cfg.rgb_pixel_mean,std=cfg.rgb_pixel_std)])
        
    def add_thorax(self, joint_coord,lshoulder_idx,rshoulder_idx):
        thorax = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord
    
    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        rgb_img = ret["image"]
        img_3d_pts = ret["img_3d_pts"]
        # 计算相对深度
        img_3d_pts[:,2] = img_3d_pts[:,2] - img_3d_pts[0,2]
        
        cam_3d_pts = ret["cam_3d_pts"]
        joints_vis = ret["joints_vis"]
         # 通过仿射变换裁剪图像
        bbox = self.get_pts_bbox(img_3d_pts,rgb_img.shape)
        img_patch, trans = self.corp_image(rgb_img,bbox,self.cfg.input_shape)
        
        for i in range(len(img_3d_pts)):
            # 仿射变换
            img_3d_pts[i, 0:2] = trans_point2d(img_3d_pts[i, 0:2], trans)
            img_3d_pts[i, 2] = ( img_3d_pts[i, 2] / (self.cfg.bbox_3d_shape[0]/2) + 1.0 ) / 2  # 归一化到0-1之间

            joints_vis[i] *= (
                            (img_3d_pts[i,0] >= 0) & \
                            (img_3d_pts[i,0] < self.cfg.input_shape[1]) & \
                            (img_3d_pts[i,1] >= 0) & \
                            (img_3d_pts[i,1] < self.cfg.input_shape[0]) & \
                            (img_3d_pts[i,2] >= 0) & \
                            (img_3d_pts[i,2] < 1)
                            )
        
        img_3d_pts[:,2] = img_3d_pts[:,2] * self.cfg.depth_dim
        img_3d_pts[:,0] = img_3d_pts[:,0] / self.cfg.input_shape[1] * self.cfg.output_shape[1]
        img_3d_pts[:,1] = img_3d_pts[:,1] / self.cfg.input_shape[0] * self.cfg.output_shape[0]
        
        rgb = self.img_transforms(img_patch.astype(np.uint8))
        
        K = torch.from_numpy(ret["KRT"][0])
        bbox = torch.from_numpy(np.asanyarray(bbox)).to(torch.float32)
        img_3d_pts = torch.from_numpy(img_3d_pts)
        cam_3d_pts = torch.from_numpy(cam_3d_pts)
        joints_vis = torch.from_numpy(ret["joints_vis"])
        # 始终除一个系数，防止key值过大
        key_value = torch.from_numpy(np.array([math.sqrt(self.cfg.bbox_3d_shape[0]*self.cfg.bbox_3d_shape[1]*K[0][0]*K[1][1]/(bbox[2]*bbox[3])) / 4000]).astype(np.float32))
        return {
            "joints_vis": joints_vis,
            'img_3d_pts': img_3d_pts,
            'cam_3d_pts': cam_3d_pts, 
            "image": rgb,
            "K": K,
            "bbox": bbox,
            "k_value": key_value
        }
        
        
    def __len__(self):
        return super().__len__()
    
    
    