import torch
from .triangulation import VolumetricTriangulationNet
from .utils.cfg import load_config
def export_triangulation_to_onnx(save_path):
    cfg = load_config("models/trianglehpe/human36m_vol_softmax.yaml")
    model = VolumetricTriangulationNet(cfg,device="cpu")
    
    images=torch.randn(*[1,4,3,256,256],dtype=torch.float32)
    proj_matricies=torch.randn(*[1, 4, 3, 4],dtype=torch.float32)
    keypoints_3d=[torch.randn(*[17, 4],dtype=torch.float32)]
    out = model(images,proj_matricies,keypoints_3d)
    
    torch.onnx.export(
        model,
        (images,proj_matricies,keypoints_3d),
        save_path,
    )
    
    
    