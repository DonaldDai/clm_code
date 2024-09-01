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
from glob import glob

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce
import pandas as pd
from pathlib import Path
from const import seq_interval, bool_interval
from sklearn.model_selection import train_test_split
# 处理的是分解后的分子结构，用于更复杂的化学反应建模，使用SMARTS标记定义了分子中的可变和不变部分。
global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")

def encode_seq(value) -> str:
    # 遍历每个区间字符串
    for interval in seq_interval:
        # 使用正则表达式提取边界
        bounds = re.findall(r'[\(\[]([^,]+),\s*([^,\)\]]+)[\)\]]', interval)
        if bounds:
            lower, upper = bounds[0]

            # 处理无穷大
            if lower == '-inf':
                lower = -math.inf
            else:
                lower = float(lower)

            if upper == 'inf':
                upper = math.inf
            else:
                upper = float(upper)

            # 检查数值是否属于当前区间
            if (lower < value <= upper) or (math.isclose(value, lower, rel_tol=1e-9) and '(' not in interval[0]):
                return interval
    return 'error'

def encode_bool(value) -> str:
    if value == 1:
        return bool_interval[0]
    elif value == -1:
        return bool_interval[1]
    elif value == 0:
        return bool_interval[2]
    return 'error'



def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=False)
    parser.add_argument("--train-ratio", "-r", help=("ration as train"), type=float,default=0.8, required=False)
    parser.add_argument("--drop-duplicated", "-d", help=("if drop the duplicated MMP "), type=int, default=0,  required=False)

    return parser.parse_args()

SEED = 42

if __name__ == "__main__":
#  constantSMILES、fromVarSMILES 和 toVarSMILES，这些列代表分子结构中的不变部分和可变部分。
    args = parse_args()
    SPLIT_RATIO = args.train_ratio
    def gen_train_data(file_path):
        dfInput=pd.read_csv(file_path)
        dfInput=dfInput.drop_duplicates(subset=['constantSMILES','fromVarSMILES','toVarSMILES'])
        dfInput=dfInput[['constantSMILES','fromVarSMILES','toVarSMILES','Value_Diff', 'main_cls', 'minor_cls', 'value_type']]
        dfInput.columns=['constantSMILES','fromVarSMILES','toVarSMILES','Delta_Value', 'main_cls', 'minor_cls', 'value_type']
        newPath=Path(file_path).parent.joinpath("train_valid_test_full.csv")   ## will be saved
        dfInput.to_csv(newPath, index=None)
        # args.input_data_path=newPath.as_posix()

        # 将数值转换成编码区间
        data = dfInput
        value_type = data['value_type'][0]
        # 判断是连续值还是bool值
        if value_type == 'seq':
            data['Delta_Value'] = data['Delta_Value'].apply(encode_seq)
        elif value_type == 'bool':
            data['Delta_Value'] = data['Delta_Value'].apply(encode_bool)
        
        # save encodeed file
        output_file = file_path.split('.csv')[0] + '_encoded.csv'
        LOG.info("Saving encoded property change to file: {}".format(output_file))
        data.to_csv(output_file, index=False)

        # split data
        train, test = train_test_split(
            data, test_size=(1-SPLIT_RATIO)/2, random_state=SEED)
        train, validation = train_test_split(train, test_size=(1-SPLIT_RATIO)/2, random_state=SEED)
        LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

        parent = uf.get_parent_dir(file_path)
        train.to_csv(os.path.join(parent, "train.csv"), index=False)
        validation.to_csv(os.path.join(parent, "validation.csv"), index=False)
        test.to_csv(os.path.join(parent, "test.csv"), index=False)

    root = '/home/yichao/zhilian/GenAICode/Data/MMPFinised/*'
    # csvFiles = glob(f"{root}/*_MMP.csv")
    # for file in csvFiles:
    #     LOG.info(f"\n=== handling {file}")
    #     gen_train_data(file)
    
    # merge train data
    combined_df = pd.DataFrame()
    trainFiles = glob(f"{root}/train.csv")
    for train_file in trainFiles:
        LOG.info(f"\n=== mergin train: {train_file}")
        df = pd.read_csv(train_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    # 保存到新的 CSV 文件，不包含索引
    combined_df.to_csv('./train.csv', index=False)

    # merge validation data
    combined_val_df = pd.DataFrame()
    valFiles = glob(f"{root}/validation.csv")
    for val_file in valFiles:
        LOG.info(f"\n=== mergin val: {val_file}")
        df = pd.read_csv(val_file)
        combined_val_df = pd.concat([combined_val_df, df], ignore_index=True)
    # 保存到新的 CSV 文件，不包含索引
    combined_val_df.to_csv('./validation.csv', index=False)
