# 3DHPE-Benchmark

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 pylib/train.py
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 pylib/train.py
```

```deploy
onnxsim --input-shape="input:1,3,640,640" tmp/mature/rtmo_s.onnx tmp/mature/rtmo_s.onnx
mnnconvert --framework ONNX --modelFile tmp/mature/rtmo_l_640x640.onnx --fp16 --MNNModel tmp/mature/rtmo_l_640x640.mnn
mnnconvert --framework ONNX --modelFile tmp/mature/rtmo_s.onnx --fp16 --saveStaticModel --MNNModel tmp/mature/rtmo_s.mnn

mnnconvert --framework ONNX --modelFile tmp/mature/dw-ll_ucoco_384x288.onnx --fp16 --MNNModel tmp/mature/dw-ll_ucoco_384x288.mnn
```
