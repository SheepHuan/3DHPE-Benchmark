

## RTMO Pose


1. Train

```bash
conda activate openmmpose
bash tools/dist_train.sh configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py 2 --work-dir /root/code/3DHPE-Benchmark/log/rtmo-l_16xb16-600e_coco-640x640 --amp

```
