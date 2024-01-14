import torch
from models.srnet.srnet import TemporalModel



def export_srnet_to_onnx(save_path):
    
    model_pos = TemporalModel(17,2,17,filter_widths = [1,1,1], causal = False, dropout = 0, channels = 1024, dense = False).eval()
    
    kp_2d = torch.randn([1,1,17,2],dtype=torch.float32)
    
    out = model_pos(kp_2d)
    
    
    torch.onnx.export(
        model_pos,
        kp_2d,
        save_path,
    )