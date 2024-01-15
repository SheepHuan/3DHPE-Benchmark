import torch
from torch import nn
from models.cpn.network import CPN101



def export_cpn_to_onnx(save_path):
    output_shape = (96, 72) #height, width
    num_class = 17

    model = CPN101(output_shape,num_class,False).eval()

    img = torch.randn([1,3,384,288],dtype=torch.float32)
    
    global_outputs, refine_output = model([img])
    
    torch.onnx.export(
        model,
        img,
        save_path,
        input_names=["img1"],
        opset_version=18
    )
    
    