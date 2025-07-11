#!/bin/bash
set -e  # 一旦出现错误就终止脚本

# 1. 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. 同步项目依赖
uv sync

# 3. 创建 data 目录
mkdir -p data

# 4. 进入 data 目录
cd data

# 5. 下载数据文件
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz

# 6. 解压数据文件
gunzip owt_train.txt.gz

# 7. 回到上一级目录
cd ..