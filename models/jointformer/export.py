from torch import nn
import torch
from models.jointformer.jointformer import JointTransformer,ErrorRefinement

class JointFormerModel(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_pos = JointTransformer(num_joints_in=17, 
                                     n_layers=4, 
                                     encoder_dropout=0.0, 
                                     d_model=64, 
                                     intermediate=True,
                                     spatial_encoding=False, 
                                     pred_dropout=0.2, 
                                     embedding_type='conv')
        
        self.model_refine = ErrorRefinement(num_joints_in=17, 
                                       n_layers=2, 
                                       d_model=256, 
                                       encoder_dropout=0.0, 
                                       pred_dropout=0.1, 
                                       spatial_encoding=False, 
                                       intermediate=False, 
                                       d_inner=1024)
        
    def forward(self,kp_2d):
        
        pred_3d, _, error_3d = self.model_pos(kp_2d)
        pred_3d = pred_3d[-1]
        error_3d = error_3d[-1]
        refine_in = torch.concat([kp_2d,pred_3d,error_3d],dim=2)
        refined_out = self.model_refine(refine_in)
        
        return refined_out
        
        
def export_joint_former_to_onnx(save_path):
    kp_2d_in = torch.randn([1,17,2],dtype=torch.float32)
    
    model = JointFormerModel().eval()
    
    out = model(kp_2d_in)
    
    torch.onnx.export(
        model,
        kp_2d_in,
        save_path,
        input_names=["2d_kp"]
    )
    
    
    
