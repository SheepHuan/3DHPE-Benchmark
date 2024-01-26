import cv2
import json
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import pickle
import copy
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord



class SingleHuman36M(Dataset):
    """
    输出：
    人体边界框图像、相机视角下的3D Pose位姿
    
    """
    
    def __init__(self,dataset_root_path,is_train=False):
        # super(self,Human36M).__init__()
        
        self.img_dir=osp.join(dataset_root_path,"images")
        self.annot_path=osp.join(dataset_root_path,"annot",f"h36m_{'train' if is_train else 'validation'}.pkl")
        self.db = self.load_db(self.annot_path)
    
        self.db = [db_rec for db_rec in self.db if not self.isdamaged(db_rec)]

        
        
    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset
        

        
    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def isdamaged(self, db_rec):
        #damaged seq
        #'Greeting-2', 'SittingDown-2', 'Waiting-1'
        if db_rec['subject'] == 9:
            if db_rec['action'] != 5 or db_rec['subaction'] != 2:
                if db_rec['action'] != 10 or db_rec['subaction'] != 2:
                    if db_rec['action'] != 13 or db_rec['subaction'] != 1:
                        return False
        else:
            return False
        return True

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        image_file = osp.join(self.img_dir,
                            db_rec['image'])
        
        bgr_img = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
        
        
        img_2d_pts = db_rec['joints_2d'].copy()
        # joints_3d = db_rec['joints_3d'].copy()
        cam_3d_pts = db_rec['joints_3d_camera'].copy()
        cam_3d_pts_normed = cam_3d_pts - cam_3d_pts[0]
        keypoint_scale = np.linalg.norm(cam_3d_pts_normed[8] - cam_3d_pts_normed[0])
        cam_3d_pts_normed /= keypoint_scale
        
        center = np.array(db_rec['center']).copy()
        joints_vis = db_rec['joints_vis'].copy()
        scale = np.array(db_rec['scale']).copy()
        
        #undistort
        camera = db_rec['camera']
        R = camera['R'].copy()
        rotation = 0
        K = np.array([
            [float(camera['fx']), 0, float(camera['cx'])], 
            [0, float(camera['fy']), float(camera['cy'])], 
            [0, 0, 1.], 
            ])
        T = camera['T'].copy()
        world_3d_pts = (R.T @ cam_3d_pts.T  + T).T
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()

        distCoeffs = np.array([float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        
        img_2d_pts = cv2.undistortPoints(img_2d_pts[:, None, :], K, distCoeffs, P=K).squeeze()
        center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()
        img_3d_pts = cam2pixel(cam_3d_pts,[K[0][0],K[1][1]],[K[0][2],K[1][2]])
        
        
        # 通过仿射变换裁剪图像
        bbox = self.get_pts_bbox(img_3d_pts)
        img_patch, trans = self.corp_image(rgb_img,bbox,[256,256])
        
        for i in range(len(img_3d_pts)):
            # 仿射变换
            img_3d_pts[i, 0:2] = trans_point2d(img_3d_pts[i, 0:2], trans)
            
            # 归一化
            img_3d_pts[i, 2] /= (2000/2.) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
            img_3d_pts[i, 2] = (img_3d_pts[i,2] + 1.0)/2. # 0~1 normalize
            joints_vis[i] *= (
                            (img_3d_pts[i,0] >= 0) & \
                            (img_3d_pts[i,0] < 256) & \
                            (img_3d_pts[i,1] >= 0) & \
                            (img_3d_pts[i,1] < 256) & \
                            (img_3d_pts[i,2] >= 0) & \
                            (img_3d_pts[i,2] < 1)
                            )
        
        return {
            "joints_vis": joints_vis,
            'img_2d_pts': img_2d_pts,
            'img_3d_pts': img_3d_pts,
            'world_3d_pts': world_3d_pts, 
            'cam_3d_pts': cam_3d_pts, 
            'cam_3d_pts_normed': cam_3d_pts_normed, 
            "image": img_patch,
            "KRT": [K,R,T]
        }

    def corp_image(self,image,bbox,target_shape):
        img = copy.deepcopy(image)
        img_height, img_width, img_channels = img.shape
        
        bb_c_x = float(bbox[0] + 0.5*bbox[2])
        bb_c_y = float(bbox[1] + 0.5*bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, target_shape[1], target_shape[0])
        
        img_patch = cv2.warpAffine(img, trans, (target_shape[1], target_shape[0]), flags=cv2.INTER_LINEAR)
        
        img_patch = img_patch[:,:,::-1].copy()
        img_patch = img_patch.astype(np.float32)
        
        return img_patch, trans

    @staticmethod
    def visualize(item,save_path):
        image = item["image"]
        points_2d =  item["img_2d_pts"]
        img_3d_pts = item["img_3d_pts"]
        
        # 绘制2D
        vis_2d = copy.deepcopy(image)
        for i in range(points_2d.shape[0]):
            vis_2d = cv2.circle(vis_2d, (int(points_2d[i][0]), int(points_2d[i][1])), 3, (0,0,255), -1)
        # cv2.imwrite(save_path,vis_2d) 
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  
        
        x,y,z = img_3d_pts[:,0],img_3d_pts[:,1],img_3d_pts[:,2]
        ax.scatter(x, y, z, c='r', marker='o')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=135, azim=90)  # 将Z轴朝向屏幕

        plt.savefig(save_path)

    def get_pts_bbox(self,coords):
        # 计算最小和最大坐标值
        min_x = np.min(coords[:, 0])
        max_x = np.max(coords[:, 0])
        min_y = np.min(coords[:, 1])
        max_y = np.max(coords[:, 1])

        # 构建边界框
        bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
        
        return bbox  

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale=1, rot=0, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


if __name__=="__main__":
    dataset_root_path="/dataset/Human3.6M/"
    
    dataset = SingleHuman36M(dataset_root_path)
    
    for idx,item in enumerate(dataset):
        os.makedirs("tmp/vis/",exist_ok=True)
        save_path=f"tmp/vis/h36m_{idx}.png"
        SingleHuman36M.visualize(item,save_path)

    