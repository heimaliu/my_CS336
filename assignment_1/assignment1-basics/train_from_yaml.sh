#!/bin/bash

# 运行 TinyStories 训练脚本
# 指定 PYTHONPATH 确保能找到 cs336_basics 包
export PYTHONPATH=$PYTHONPATH:$(pwd)

python cs336_basics/train_tinystories_lm.py --config cs336_basics/config.yaml
