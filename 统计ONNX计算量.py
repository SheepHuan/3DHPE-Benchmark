import numpy as np

def profile_model(modelpath):
    import onnx_tool
    m = onnx_tool.Model(modelpath)


    m.graph.shape_infer({'data': np.zeros((1,1, 17,2))})  # update new resolution
    m.graph.profile()
    m.graph.print_node_map()  # remove ops from the profile
    

# path="tmp/onnx/mobileposenet3d.onnx"
# path="tmp/jointformer.onnx"
path="tmp/srnet.onnx"
profile_model(path)