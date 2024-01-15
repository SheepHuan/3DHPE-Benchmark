from models.jointformer.export import export_joint_former_to_onnx
from models.srnet.export import export_srnet_to_onnx
from models.d3dp.export import export_d3dp_to_onnx
from models.cpn.export import export_cpn_to_onnx
from models.maskrcnn.export import export_maskrcnn_to_onnx


if __name__=="__main__":
    # export_joint_former_to_onnx("tmp/jointformer.onnx")
    # export_srnet_to_onnx("tmp/srnet.onnx")
    # export_d3dp_to_onnx("tmp/d3dp.onnx")
    # export_cpn_to_onnx("tmp/cpn.onnx")
    export_maskrcnn_to_onnx("tmp/maskrcnn.onnx")