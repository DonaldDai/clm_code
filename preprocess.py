"""
Preprocess
- encode property change
- build vocabulary
- split data into train, validation and test
"""
import os
import argparse
import pickle

import preprocess.vocabulary as mv
import preprocess.data_preparation as pdp
import configuration.config_default as cfgd
import utils.log as ul
import utils.file as uf
import preprocess.property_change_encoder as pce
import pandas as pd
from pathlib import Path
# 处理的是分解后的分子结构，用于更复杂的化学反应建模，使用SMARTS标记定义了分子中的可变和不变部分。
global LOG
LOG = ul.get_logger("preprocess", "experiments/preprocess.log")


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Preprocess: encode property change and build vocabulary")

    parser.add_argument("--input-data-path", "-i", help=("Input file path"), type=str, required=True)
    parser.add_argument("--train-ratio", "-r", help=("ration as train"), type=float,default=0.8, required=False)
    parser.add_argument("--drop-duplicated", "-d", help=("if drop the duplicated MMP "), type=int, default=0,  required=False)

    return parser.parse_args()


if __name__ == "__main__":
#  constantSMILES、fromVarSMILES 和 toVarSMILES，这些列代表分子结构中的不变部分和可变部分。
    args = parse_args()
    if args.drop_duplicated:
        print("Duplicated ['constantSMILES','fromVarSMILES','toVarSMILES'] will be dropped!")
        dfInput=pd.read_csv(args.input_data_path)
        dfInput=dfInput.drop_duplicates(subset=['constantSMILES','fromVarSMILES','toVarSMILES'])
        dfInput=dfInput[['constantSMILES','fromVarSMILES','toVarSMILES','Pk_Diff']]
        dfInput.columns=['constantSMILES','fromVarSMILES','toVarSMILES','Delta_pki']
        newPath=Path(args.input_data_path).parent.joinpath("train_valid_test_full.csv")   ## will be saved
        dfInput=dfInput.to_csv(newPath, index=None)
        args.input_data_path=newPath.as_posix()

    # encode property change without adding property name
    # 获取编码的区间和用来找区间的辅助数据结构 <'pki', str[] <number, str>>
    property_change_encoder = pce.encode_property_change(args.input_data_path)

    # add property name before property change; save to file
    property_condition = []
    for property_name in cfgd.PROPERTIES:
        if property_name == 'pki':
            intervals, _ = property_change_encoder[property_name]
            property_condition.extend(intervals)
            
        elif property_name == 'qed':
            intervals, _ = property_change_encoder[property_name]
            property_condition.extend(intervals)
            
        elif property_name == 'sa':
            intervals, _ = property_change_encoder[property_name]
            property_condition.extend(intervals)
     
    LOG.info("Property condition tokens: {}".format(len(property_condition)))
    # 将数值转换成编码区间
    encoded_file = pdp.save_df_property_encoded(args.input_data_path, property_change_encoder, LOG)
    LOG.info("Building vocabulary")
    tokenizer = mv.SMILESTokenizer()
    # 获取所有SMILES分子式列表
    smiles_list = pdp.get_smiles_list(args.input_data_path)  ## updated for constant SMILES
    # 将SMILES和属性编码传入 属性编码直接转换成数字，SMILES token化后转换成数字
    vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer, property_condition=property_condition)
    tokens = vocabulary.tokens()
    LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)

    # Save vocabulary to file
    # 保存词典
    parent_path = uf.get_parent_dir(args.input_data_path)
    output_file = os.path.join(parent_path, 'vocab.pkl')
    with open(output_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)
    LOG.info("Save vocabulary to file: {}".format(output_file))
# 通过 args.train_ratio 参数来设置训练集的比例
    # Split data into train, validation, test
    # 所有数据分割成训练集和测试集，从训练集中再分出验证集 最后将所有文件保存
    train, validation, test = pdp.split_data(encoded_file, args.train_ratio, LOG)

