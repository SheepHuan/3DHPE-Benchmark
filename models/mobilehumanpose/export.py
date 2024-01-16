import torch
from .model import get_pose_net


def export_mobileposenet3d_to_onnx(save_path,shape=[1,3,256,256]):
    model = get_pose_net(
        'LPSKI',False,17
    )
    model.eval()
    
    inputs=torch.randn(*shape,dtype=torch.float32)
    
    out = model(inputs)
    
    torch.onnx.export(
        model,
        inputs,
        save_path,
        input_names=["inputs"]
    )
