import argparse

import os
import torch.distributed as dist
import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
# from trainer.seq2seq_trainer import Seq2SeqTrainer
from common.utils import Data_Type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opts.train_opts_transformer(parser)
    parser.add_argument("--bar", type=bool, default=False)
    parser.add_argument("--data-type", type=str, default=Data_Type.frag.value)
    opt = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    trainer = TransformerTrainer(opt, local_rank, rank, world_size)
        
    # elif opt.model_choice == 'seq2seq':
    #     trainer = Seq2SeqTrainer(opt)
    print(f"Starting training on rank {rank} out of {world_size} processes | bar {opt.bar} | data_type {opt.data_type}")
    trainer.train(opt)
