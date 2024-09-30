"""
Preprocess
- encode property change
- build vocabulary
- split data into train, validation and test
"""
import os
import argparse
import pickle
import re
import math

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce
import pandas as pd
from pathlib import Path
from glob import glob
import generate as gn
from const import seq_interval, bool_interval
# 处理的是分解后的分子结构，用于更复杂的化学反应建模，使用SMARTS标记定义了分子中的可变和不变部分。
global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")

def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=False)
    parser.add_argument("--train-ratio", "-r", help=("ration as train"), type=float,default=0.8, required=False)
    parser.add_argument("--drop-duplicated", "-d", help=("if drop the duplicated MMP "), type=int, default=0,  required=False)

    return parser.parse_args()


if __name__ == "__main__":
#  constantSMILES、fromVarSMILES 和 toVarSMILES，这些列代表分子结构中的不变部分和可变部分。
    args = parse_args()

    def record_vocal(vocabulary, file):
        dfInput=pd.read_csv(file)
        if len(dfInput) < 1:
            return
        LOG.info("===finish reading")
        # add property name before property change; save to file
        property_condition = []
        # 添加main_cls
        property_condition.append(dfInput['main_cls'][0])
        # 添加minor_cls
        property_condition.append(dfInput['minor_cls'][0])
        # 添加靶点信息
        target_word_list = list(dfInput['target_name'][0])
        
        dfInput=dfInput.drop_duplicates(subset=['constantSMILES','fromVarSMILES','toVarSMILES'])
        dfInput=dfInput[['constantSMILES','fromVarSMILES','toVarSMILES','Value_Diff', 'main_cls', 'minor_cls', 'value_type', 'target_name']]
        dfInput.columns=['constantSMILES','fromVarSMILES','toVarSMILES','Delta_Value', 'main_cls', 'minor_cls', 'value_type', 'target_name']
        # newPath=Path(args.input_data_path).parent.joinpath("train_valid_test_full.csv")   ## will be saved
        # dfInput=dfInput.to_csv(newPath, index=None)
        # args.input_data_path=newPath.as_posix()

        
        
        # LOG.info("Property condition tokens: {}".format(len(property_condition)))
        # # 将数值转换成编码区间
        # encoded_file = save_df_property_encoded(args.input_data_path, LOG)
        LOG.info("Building vocabulary")
        tokenizer = mv.SMILESTokenizer()
        # 获取所有SMILES分子式列表
        smiles_list = pd.unique(dfInput[['constantSMILES', 'fromVarSMILES', 'toVarSMILES']].values.ravel('K'))
        # 将SMILES和属性编码传入 属性编码直接转换成数字，SMILES token化后转换成数字
        tokens = set()
        for smi in smiles_list:
            tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

        vocabulary.update(target_word_list)
        vocabulary.update(sorted(tokens))
        vocabulary.update(property_condition)
        

    vocabulary = mv.Vocabulary()
    # pad=0, start=1, end=2, default_key for key error
    vocabulary.update(["*", "^", "$", "default_key"])
    interval_token = []
    # 改为固定区间
    # 连续值区间
    interval_token.extend(seq_interval)
    # 布尔值区间
    interval_token.extend(bool_interval)
    csvFiles = glob(f"/home/yichao/zhilian/GenAICode/Data/MMPFinised/*/*_MMP.csv")
    # 记录smiles main_cls minor_cls
    for idx, file in enumerate(csvFiles):
        LOG.info(f"===handling {idx} {file}")
        record_vocal(vocabulary, file)
        # if idx > 500:
        #     break
    # Save vocabulary to file
    # 保存词典
    # parent_path = uf.get_parent_dir(args.input_data_path)
    # output_file = os.path.join(parent_path, 'vocab.pkl')
    # for random smiles
    vocabulary.update(interval_token)
    if "8" not in vocabulary.tokens():
        vocabulary.update(["8"])
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)
    output_file = './vocab.pkl'
    with open('./vocab.pkl', 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))