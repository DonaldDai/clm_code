#!/bin/bash
#SBATCH --job-name=c_raw_frag_clm_pretrain    # 作业名称
#SBATCH --nodes=8                  # 请求的节点数
#SBATCH --ntasks-per-node=1         # 每个节点上的任务数
#SBATCH --output=./logs/clm_pretrain_%j.out # 标准输出文件
#SBATCH --error=./logs/clm_pretrain_%j.err  # 标准错误文件
#SBATCH -p gpu-a100                 # 指定分区
#SBATCH -t 48:00:00                 # 时间限制
#SBATCH --mail-user=daizhilian@hotmail.com
#SBATCH --mail-type=begin               
#SBATCH --mail-type=end

# 初始化 Conda 环境
source /work/09735/yichao/ls6/miniconda/etc/profile.d/conda.sh

# 激活你的 Conda 环境
conda activate drugassist-jay

# 切换到特定的工作目录
cd /work/09735/yichao/ls6/zhilian/clm_code

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 获取主节点的地址
MASTER_ADDR=$(hostname)

# 执行PyTorch分布式训练命令
# 端口最好使用随机的
t="frag"
# continue
srun bash -c "torchrun --nproc_per_node=3 --nnodes=8 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=39688 train.py --model-choice transformer --data-type ${t} --batch-size 384 --num-epoch 200  --data-path /work/09735/yichao/ls6/zhilian/clm_code  --save-directory /work/09735/yichao/ls6/zhilian/clm_code/raw_pretrain_${t} --starting-epoch 16 --pretrain-path /work/09735/yichao/ls6/zhilian/clm_code/raw_pretrain_${t}"
# srun bash -c "torchrun --nproc_per_node=3 --nnodes=8 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=39688 train.py --model-choice transformer --data-type ${t} --batch-size 384 --num-epoch 200  --data-path /work/09735/yichao/ls6/zhilian/clm_code  --save-directory /work/09735/yichao/ls6/zhilian/clm_code/raw_pretrain_${t}"
