from .single_h36m import SingleHuman36M
import numpy as np
import torchvision.transforms as transforms


class PoseNet3dDataset(SingleHuman36M):
    
    def __init__(self):
        super(PoseNet3dDataset,self).__init__()
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
        
    def add_thorax(self, joint_coord,lshoulder_idx,rshoulder_idx):
        thorax = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord
    
    def __getitem__(self, idx):
        ret = super().__getitem__()
        
       
        rgb = ret["image"]
        img_3d_pts = ret["img_3d_pts"]
        cam_3d_pts = ret["cam_3d_pts"]
        root_cam = cam_3d_pts[0]
        relative_img_3d_pts= img_3d_pts[:,2] - root_cam
        
        bbox = self.get_pts_bbox(img_3d_pts)
        # 根据bbox裁剪图像，然后resize成256x256
        
        
        return {
            "image": self.img_transforms(rgb),
            "img_3d_pts": relative_img_3d_pts,
            "cam_3d_root_pt": root_cam,
            "bbox": bbox,
            "joints_vis": ret["joints_vis"],
        }
        
        
        
    
    
    