# A-PFG

This is an official implementation for AAAI 2023 paper "Learning to Generate an Unbiased Scene Graph by Using Attribute-Guided Predicate Features".

## Installation

See [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

See [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Training

Our experiments are conducted on 2 NVIDIA GeForce RTX 3090, If you want to run it on your own device, please refer to [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

The following command can be used to train the models:
```
# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10100 --nproc_per_node=2 tools/APFG_main.py --config-file "./configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --my_opts "./maskrcnn_benchmark/APFG/APFG_preds.yaml"

# 1 GPUs
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10080 --nproc_per_node=1 tools/APFG_main.py --config-file "./configs/e2e_relation_X_101_32_8_FPN_1x.yaml" --my_opts "./maskrcnn_benchmark/APFG/APFG_preds.yaml"
```
Please modify the path in `./maskrcnn_benchmark/APFG/APFG_preds.yaml` to the path on your own device.

If you have any questions, please contact me (`wlei0618@foxmail.com`).

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
