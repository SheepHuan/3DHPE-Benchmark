import torch
from models.d3dp.diffusionpose import D3DP
from models.d3dp.cfg import parse_args
args = parse_args()



def export_d3dp_to_onnx(save_path,p_num=1):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    
    # sampling_timesteps 表示迭代次数 
    model_pos = D3DP(joints_left, joints_right,  is_train=False, num_proposals=5, sampling_timesteps=5,timestep=1000).eval()
    kp_2d = torch.randn([1,p_num,17,2],dtype=torch.float32)
    
    out = model_pos(kp_2d)
    
    torch.onnx.export(
        model_pos,
        kp_2d,
        save_path,
        input_names=["2d_kp"]
    )
    
    