import numpy as np

def profile_model_maskrcnn(modelpath):
    import onnx_tool
    m = onnx_tool.Model(modelpath)


    m.graph.shape_infer({'image.1': np.random.rand(3, 300,400).astype(np.float32),
                         'image.5': np.random.rand(3, 300,400).astype(np.float32),})  # update new resolution
    m.graph.profile()
    m.graph.print_node_map()  # remove ops from the profile
    

# path="tmp/onnx/mobileposenet3d.onnx"
# path="tmp/jointformer.onnx"
path="tmp/maskrcnn.onnx"
profile_model_maskrcnn(path)