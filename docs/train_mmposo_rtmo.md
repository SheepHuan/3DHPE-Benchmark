# Train MMPose

以 rtmo 为例

## 1 准备数据集

[mmpose 文档地址](https://mmpose.readthedocs.io/zh-cn/latest/user_guides/prepare_datasets.html)

### 1.1 准备 coco 数据集

暂时掠过

### 1.1.1 浏览数据集

[mmpose 文档地址](https://mmpose.readthedocs.io/zh-cn/latest/user_guides/prepare_datasets.html#id5)

```bash
conda activate openmmpose

cd mmpose

python tools/misc/browse_dataset.py configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py --not-show --phase val --mode transformed --show-interval 1 --output-dir /root/code/3DHPE-Benchmark/tmp/vis_coco


python tools/misc/browse_dataset.py configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py --not-show --phase val --mode original --show-interval 1 --output-dir /root/code/3DHPE-Benchmark/tmp/vis_coco

rm -rf /root/code/3DHPE-Benchmark/tmp/vis_coco
```

### 1.2 自定义 human3.6m 2D 数据集

## 训练

### 单卡训练

[mmpose 文档](https://mmpose.readthedocs.io/zh-cn/latest/user_guides/train_and_test.html#id3)

```bash
conda activate openmmpose

CUDA_VISIBLE_DEVICES="0,1" python tools/train.py configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py --work-dir /root/code/3DHPE-Benchmark/log/rtmo-l_16xb16-600e_coco-640x640 --auto-scale-lr --amp


```

### 多卡分布式训练

```bash

conda activate openmmpose

bash tools/dist_train.sh \
configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py \
2 \
--work-dir /root/code/3DHPE-Benchmark/log/rtmo-l_16xb16-600e_coco-640x640 \
--amp

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 pylib/train.py


CUDA_VISIBLE_DEVICES=0,1 python -m torchrun --nproc_per_node=2 pylib/train.py
```

## 自定义评估
