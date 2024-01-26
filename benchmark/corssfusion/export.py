import torch
from .models.pose_resnet import get_pose_net
from .models.multiview_pose_resnet import get_multiview_pose_net
from .config import update_config,config
from .multiviews.pictorial_cuda import rpsm,compute_grid,compute_pairwise,compute_unary_term,infer,get_loc_from_cube_idx,recursive_infer
import numpy as np
def routing(raw_features, aggre_features, is_aggre, meta):
    if not is_aggre:
        return raw_features

    output = []
    for r, a, m in zip(raw_features, aggre_features, meta):
        view = torch.zeros_like(a)
        batch_size = a.size(0)
        for i in range(batch_size):
            s = m['source'][i]
            view[i] = a[i] if s == 'h36m' else r[i]
        output.append(view)
    return output




def export_corss_fusion(path):
    update_config("models/corssfusion/resnet152_256_fusion.yaml")
    backbone=get_pose_net(config,is_train=False)
    kp2d_model = get_multiview_pose_net(backbone,config).eval()
 
    inputs = [torch.randn(*[1,3,256,256],dtype=torch.float32) for _ in range(4)]

    out = kp2d_model(inputs)
    
    torch.onnx.export(
        kp2d_model,
        inputs,
        path,
    )
    

