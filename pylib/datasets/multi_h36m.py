import cv2
import json
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import pickle
import copy
from .single_h36m import SingleHuman36M

class MultiViewHuman36M(SingleHuman36M):
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

        
        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        # self.human_bbox_root_dir = osp.join('..', 'data', 'Human36M', 'bbox_root', 'bbox_root_human36m_output.json')
        
    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset
        
    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i
            
        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        # if self.is_train:
        #     if cfg.DATASETS.H36M.TRAIN_SAMPLE:
        #         filtered_grouping = filtered_grouping[::cfg.DATASETS.H36M.TRAIN_SAMPLE]
        # else:
        #     if cfg.DATASETS.H36M.TEST_SAMPLE:
        #         filtered_grouping = filtered_grouping[::cfg.DATASETS.H36M.TEST_SAMPLE]

        return filtered_grouping
        
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
        
        image_data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        
        
        joints = db_rec['joints_2d'].copy()
        joints_3d = db_rec['joints_3d'].copy()
        joints_3d_camera = db_rec['joints_3d_camera'].copy()
        joints_3d_camera_normed = joints_3d_camera - joints_3d_camera[0]
        keypoint_scale = np.linalg.norm(joints_3d_camera_normed[8] - joints_3d_camera_normed[0])
        joints_3d_camera_normed /= keypoint_scale
        
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
        world3d = (R.T @ joints_3d_camera.T  + T).T
        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = -R @ T.squeeze()

        distCoeffs = np.array([float(i) for i in [camera['k'][0], camera['k'][1], camera['p'][0], camera['p'][1], camera['k'][2]]])
        
        joints = cv2.undistortPoints(joints[:, None, :], K, distCoeffs, P=K).squeeze()
        center = cv2.undistortPoints(np.array(center)[None, None, :], K, distCoeffs, P=K).squeeze()
        
        return {
            'points_2d': joints,
            'points_3d': world3d, 
            'camera_points_3d': joints_3d_camera, 
            'normed_points_3d': joints_3d_camera_normed, 
            'scale': keypoint_scale,
            "image": image_data_numpy
        }

    def __len__(self):
        return len(self.db)

    @staticmethod
    def visualize(item,save_path):
        image = item["image"]
        points_2d =  item["points_2d"]
        print(1)


if __name__=="__main__":
    dataset_root_path="/dataset/Human3.6M/"
    
    dataset = Human36M(dataset_root_path)
    
    for idx,item in enumerate(dataset):
        Human36M.visualize(item,save_path="tmp/")
    
    