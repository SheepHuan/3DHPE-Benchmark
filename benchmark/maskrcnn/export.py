import torch
import torchvision.models as models



def export_maskrcnn_to_onnx(save_path):
    model = models.detection.maskrcnn_resnet50_fpn().eval()
    

    inputs = torch.randn(*[1,3,256,256],dtype=torch.float32)
    trace_model =  torch.jit.trace(model,inputs)
    
    # out = model(inputs)
    
    torch.onnx.export(
        trace_model,
        inputs,
        save_path,
        input_names=["inputs"]
    )
