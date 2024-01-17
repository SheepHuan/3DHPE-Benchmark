import numpy as np
import onnx_tool



def proile_cpn(path):
    onnx_tool.model_profile(path, {'img': np.random.rand(1,3, 256,256).astype(np.float32)},
                            savenode='tmp/cpn.txt')

def proile_maskrcnn(path):
    from onnx_tool.node import _get_shape
    @onnx_tool.NODE_REGISTRY.register()
    class GreaterOrEqualNode(onnx_tool.Node):
        # you can implement either shape_infer(faster) or value_infer.
        # it's not necessary to implement both
        # def shape_infer(self, intensors: []):
        #     # if you know how to calculate shapes of this op, you can implement shape_infer
        #     return [_get_shape(intensors[1])]

        # for upgrade of node_profilers.py, node_profilers.py's 'infer_shape' method should be placed
        # as 'value_infer' method here, and do not create this class' 'shape_infer' method.
        def value_infer(self, intensors: [],*args):
            # if you don't know how to calculate the shapes of this op, you can implement value_infer.
            # shape1 = intensors[1].shape
            # outtensor = intensors[0][:, :, :shape1[2], :shape1[3]]
            return []

        def profile(self, intensors: [], outtensors: []):
            macs = 0
            # accumulate macs here
            # this node has no calculation
            return macs
    
    @onnx_tool.NODE_REGISTRY.register()
    class IfNode(onnx_tool.Node):
        # you can implement either shape_infer(faster) or value_infer.
        # it's not necessary to implement both
        def shape_infer(self, intensors,otensors):
            # if you know how to calculate shapes of this op, you can implement shape_infer
            return otensors.shape

        # for upgrade of node_profilers.py, node_profilers.py's 'infer_shape' method should be placed
        # as 'value_infer' method here, and do not create this class' 'shape_infer' method.
        def value_infer(self, intensors,otensors):
            # if you don't know how to calculate the shapes of this op, you can implement value_infer.
            # shape1 = intensors[1].shape
            # outtensor = intensors[0][:, :, :shape1[2], :shape1[3]]
            return otensors

        def profile(self, intensors: [], outtensors: []):
            macs = 0
            # accumulate macs here
            # this node has no calculation
            return macs    
    
    onnx_tool.model_profile(path, {'inputs': np.random.rand(1,3, 256,256).astype(np.float32)},
                            savenode='tmp/maskrcnnn.txt',)

def profile_d3dp(path):
    onnx_tool.model_profile(path, {'2d_kp': np.random.rand(1,1, 17,2).astype(np.float32)},
                            savenode='tmp/d3dp.txt')

def profile_jointformer(path):
    onnx_tool.model_profile(path, {'2d_kp': np.random.rand(1,17,2).astype(np.float32)},
                            savenode='tmp/jointformer.txt')

def profile_srnet(path):
    onnx_tool.model_profile(path, {'2d_kp': np.random.rand(1,1,17,2).astype(np.float32)},
                            savenode='tmp/srnet.txt') 

def profile_posenet(path):
    onnx_tool.model_profile(path, {'inputs': np.random.rand(1,3,256,256).astype(np.float32)},
                            savenode='tmp/posenet3d.txt') 

def profile_mobileposenet3d(path):
    onnx_tool.model_profile(path, {'inputs': np.random.rand(1,3,256,256).astype(np.float32)},
                            savenode='tmp/mobileposener3d.txt') 
    
def profile_tr(path):
    # images=torch.randn(*[1,4,3,256,256],dtype=torch.float32)
    # proj_matricies=torch.randn(*[1, 4, 3, 4],dtype=torch.float32)
    # keypoints_3d=[torch.randn(*[17, 4],dtype=torch.float32)]
    # out = model(images,proj_matricies,keypoints_3d)
    onnx_tool.model_profile(path, {
        'images': np.random.rand(1,2,3,256,256).astype(np.float32),
        'proj_matricies': np.random.rand(1, 2, 3, 4).astype(np.float32),
         'keypoints_3d': [np.random.rand(17,4).astype(np.float32)],
        
        },savenode='tmp/tr.txt') 
    
def profile_yolov6(path):
    onnx_tool.model_profile(path, {
        'images': np.random.rand(1,3,256,256).astype(np.float32),
        },savenode='tmp/yolov6.txt') 

def profile_crossfusion(path):
    onnx_tool.model_profile(path, {
        'images': np.random.rand(2,3,256,256).astype(np.float32),
        },savenode='tmp/crossfusion.txt') 

if __name__=="__main__":
    
    # proile_cpn("tmp/cpn.onnx")
    # profile_d3dp("tmp/d3dp.onnx")
    # profile_jointformer("tmp/jointformer.onnx")
    # profile_srnet("tmp/srnet.onnx")
    # profile_posenet("tmp/posenet3d.onnx")
    # profile_mobileposenet3d("tmp/mobileposener3d.onnx")
    profile_tr("tmp/tr.onnx")
    
    # proile_maskrcnn("tmp/maskrcnn.onnx")
    # profile_yolov6("tmp/yolov6l.onnx")
    # profile_crossfusion("tmp/crossfusion.onnx")