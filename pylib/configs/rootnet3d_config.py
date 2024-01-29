from .config import Config

class RootNetConfig(Config):
    resnet_type = 152
    joint_num = 17
    depth_dim = 64
    input_shape=[256,256]
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    
    model_name = ""
    
    def __init__(self) -> None:
        super().__init__()