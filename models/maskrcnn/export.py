import torchvision
import torch

def export_maskrcnn_to_onnx(save_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    x = [torch.rand(3, 300, 400),torch.rand(3, 300, 400)]
    predictions = model(x)
    
    # optionally, if you want to export the model to ONNX:
    torch.onnx.export(model, x, save_path, opset_version = 18)
