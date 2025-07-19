#!/bin/bash
# set -e  # 一旦出现错误就终止脚本

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
wget https://huggingface.co/datasets/irelandoldpig/Stanford_CS336_25Spring_Data/resolve/main/owt_train.npy
wget https://huggingface.co/datasets/irelandoldpig/Stanford_CS336_25Spring_Data/resolve/main/owt_valid.npy
wget https://huggingface.co/datasets/irelandoldpig/Stanford_CS336_25Spring_Data/resolve/main/TinyStoriesV2-GPT4-train.npy
wget https://huggingface.co/datasets/irelandoldpig/Stanford_CS336_25Spring_Data/resolve/main/TinyStoriesV2-GPT4-valid.npy

# 6. 回到上一级目录
cd ..

# 7. wandb
uv run wandb login

# 8. 更新apt
# apt update
# apt  -y install tmux nvtop vim
# source $HOME/.local/bin/env

chmod +x script/*.sh