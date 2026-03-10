#!/bin/bash

# 设置镜像站地址
HF_ENDPOINT="https://hf-mirror.com"

# 创建并进入目录
mkdir -p data
cd data

echo "正在从镜像站 $HF_ENDPOINT 下载文件..."

# 定义下载函数，优先使用 curl 并处理跳转
download_file() {
    local url=$1
    local filename=$2
    echo "下载 $filename ..."
    curl -L -C - -o "$filename" "$HF_ENDPOINT/datasets/$url"
}

# 1. TinyStories 数据集
download_file "roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" "TinyStoriesV2-GPT4-train.txt"
download_file "roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" "TinyStoriesV2-GPT4-valid.txt"

# 2. OWT Sample 数据集 (压缩包)
download_file "stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz" "owt_train.txt.gz"
if [ -f "owt_train.txt.gz" ]; then
    echo "解压 owt_train.txt.gz ..."
    gunzip -f owt_train.txt.gz
fi

download_file "stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz" "owt_valid.txt.gz"
if [ -f "owt_valid.txt.gz" ]; then
    echo "解压 owt_valid.txt.gz ..."
    gunzip -f owt_valid.txt.gz
fi

echo "任务完成！"
