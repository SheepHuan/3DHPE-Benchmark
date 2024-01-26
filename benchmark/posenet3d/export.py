import torch
from .model import get_pose_net
from .config import cfg
def export_posenet3d_to_onnx(save_path,shape=[1,3,256,256]):
    model = get_pose_net(cfg,False,17)
    model.eval()
    
    images=torch.randn(*[1,4,3,256,256],dtype=torch.float32)
    proj_matricies=torch.randn(*[1, 4, 3, 4],dtype=torch.float32)
    keypoints_3d=[torch.randn(*[17, 4],dtype=torch.float32)]
    out = model(images,proj_matricies,keypoints_3d)
    
    torch.onnx.export(
        model,
        [images,proj_matricies],
        save_path,
        export_params=True,
        input_names=["input"],
        output_names=["coords"],
        opset_version=16,
    )
