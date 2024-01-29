from .config import Config
import torch
import os

class PosenetConfig(Config):
    resnet_type = 152
    joint_num = 17
    depth_dim = 64
    input_shape=[256,256]
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    
    
    model_name=""
    
    def __init__(self) -> None:
        super().__init__()
        
    
    @staticmethod
    def loss_func(model,data,device):
        inputs = {
            "img": data["image"].to(device),
            # "bbox": data["bbox"].to(device),
            # "K": data["K"].to(device),
            # "key_value": data["k_value"].to(device)
        }
        target = {
            "img_3d_pts": data["img_3d_pts"].to(device),
            "joints_vis": data["joints_vis"].to(device),
        }
    
        loss_coord = model(**inputs, target=target)
        loss = loss_coord.mean()
        return loss
    
    
    @staticmethod
    def metric_func(iter,metrics,model,data,device):
        inputs = {
            "img": data["image"].to(device),
            # "bbox": data["bbox"].to(device),
            # "K": data["K"].to(device),
            # "key_value": data["k_value"].to(device)
        }
        target = {
            "img_3d_pts": data["img_3d_pts"].to(device),
            "joints_vis": data["joints_vis"].to(device),
        }
    
        pred_img_coord = model(**inputs, target=target)
        if "mpjpe" not in metrics.keys():
            metrics["mpjpe"]=[]
            
        gt = data["img_3d_pts"].to(device)       
        
        # 应该转换为cam再去计算相关的结果
        
        mpjpe = torch.mean(torch.sqrt(torch.sum((pred_img_coord - gt)**2,1)))
        metrics["mpjpe"].append(mpjpe.detach().item())
        print(mpjpe.detach().item())
        
        # 绘制2D,选择batch为1的
        n,k,c = pred_img_coord.shape
        if not (iter % 10 ==0 and iter!=0):
            return metrics
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        img_3d_pts = data["img_3d_pts"]
        pred_img_3d_pts = pred_img_coord.detach().cpu().numpy()
        os.makedirs("tmp/vis",exist_ok=True)
        for i in range(n):
            save_path=f"tmp/vis/i_{iter}_b{i}.png"

            plt.clf()
            fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
            # 第一个子图
            x1, y1, z1 = img_3d_pts[i, :, 0], img_3d_pts[i, :, 1], img_3d_pts[i, :, 2]
            ax1.scatter(x1, y1, z1, c='r', marker='o')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.view_init(elev=135, azim=90)

            # 第二个子图
            x2, y2, z2 = pred_img_3d_pts[i, :, 0], pred_img_3d_pts[i, :, 1], pred_img_3d_pts[i, :, 2]
            ax2.scatter(x2, y2, z2, c='b', marker='o')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.view_init(elev=135, azim=90)
            
            plt.savefig(save_path)
        
        
        return metrics