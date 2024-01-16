from models.jointformer.export import export_joint_former_to_onnx
from models.srnet.export import export_srnet_to_onnx
from models.d3dp.export import export_d3dp_to_onnx
from models.cpn.export import export_cpn_to_onnx
from models.posenet3d.export import export_posenet3d_to_onnx
from models.mobilehumanpose.export import export_mobileposenet3d_to_onnx
from models.maskrcnn.export import export_maskrcnn_to_onnx
from models.trianglehpe.export import export_triangulation_to_onnx


if __name__=="__main__":
    # export_joint_former_to_onnx("tmp/jointformer.onnx")
    # export_srnet_to_onnx("tmp/srnet.onnx")
    # export_d3dp_to_onnx("tmp/d3dp.onnx")
    # export_cpn_to_onnx("tmp/cpn.onnx")
    # export_maskrcnn_to_onnx("tmp/maskrcnn.onnx")
    # export_posenet3d_to_onnx("tmp/posenet3d.onnx")
    # export_mobileposenet3d_to_onnx("tmp/mobileposener3d.onnx")
    # export_maskrcnn_to_onnx("tmp/maskrcnn.onnx")
    export_triangulation_to_onnx("tmp/tr.onnx")