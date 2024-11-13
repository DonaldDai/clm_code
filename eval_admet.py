import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import wandb
from eval_utils import sample_data
import subprocess
import math
from DeepPurpose import utils, CompoundPred



start_time = time.time()
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

# hyper params
EPOCH = 20
MODEL_EPOCH = 50
DRUG_ENCODDING = 'MPNN'
PART_N = 5
DRUG_BIN = f'/work/09735/yichao/ls6/miniconda/envs/drugassist-jay/bin/python'
NUMS = [1]
OVERWRITE = True
for i in range(5, 16, 5):
  NUMS.append(i)
for i in range(20, 101, 20):
  NUMS.append(i)
max_num = max(NUMS)
print(f'USING NUMS: {NUMS}, max num: {max_num}')
run = wandb.init(
    project="CLM",
    notes=f"eval ADMET {formatted_time}| EPOCH {EPOCH}",
    tags=["eval", "ADMET", f"epoch{EPOCH}"],
    name=f"eval ADMET {formatted_time}| EPOCH {EPOCH}"
)
artifact = wandb.Artifact('ADMET_table', type='dataset')

def draw_seq(x, y, z):
  plt.figure(figsize=(10, 5))
  plt.plot(x, y, label='spec_acc', marker='o', linestyle='-', color='lightcoral', markerfacecolor='peachpuff')
  plt.plot(x, z, label='dir_acc', marker='o', linestyle='-', color='lightblue', markerfacecolor='peachpuff')
  plt.title('Accuracy Rates by Sample Count')
  plt.xlabel('Sample Count')
  plt.ylabel('Accuracy Rate')
  plt.grid(True, linestyle='--', color='grey', alpha=0.5)  # 使用更柔和的网格线
  return plt

def draw_bool(*args):
  plt.figure(figsize=(10, 5))
  plt.plot(*args, marker='o', linestyle='-', color='lightcoral', markerfacecolor='peachpuff')
  plt.title('Accuracy Rates by Sample Count')
  plt.xlabel('Sample Count')
  plt.ylabel('Accuracy Rate')
  plt.grid(True, linestyle='--', color='grey', alpha=0.5)  # 使用更柔和的网格线
  return plt

def parse_seq_num(delta_value_str):
  n1_str, n2_str = delta_value_str[1:-1].split(',')
  return float(n1_str), float(n2_str)

def stat_seq(origin, pred_y, delta_value):
  dir_acc_num = 0
  spec_acc_num = 0
  n1, n2 = parse_seq_num(delta_value)
  n_left = origin + n1
  n_right = origin + n2
  for y in pred_y:
    if y > n_left and y <= n_right:
      spec_acc_num = spec_acc_num + 1
    if (n1 < 0 or n2 < 0) and y - origin <= 0:
      dir_acc_num = dir_acc_num + 1
    elif (n1 > 0 or n2 > 0) and y - origin > 0:
      dir_acc_num = dir_acc_num + 1
  return dir_acc_num, spec_acc_num

def parse_bool_num(delta_value_str):
  n1_str, n2_str = delta_value_str.split('-')
  return int(n1_str), int(n2_str)

def get_pred_cls(num):
  if num > 0.8:
    return 1
  return 0

def stat_bool(origin, pred_y, delta_value):
  acc_num = 0
  _, n2 = parse_bool_num(delta_value)
  for y in pred_y:
    if get_pred_cls(y) == n2:
      acc_num = acc_num + 1
  return acc_num

CAT = {
    'absorption': ['lipophilicity_astrazeneca', 'hydrationfreeenergy_freesolv', 'solubility_aqsoldb'],
    'distribution': ['bbb_martins']
}
# for admet属性
for main, minor_list in CAT.items():
  for minor in minor_list:
    model = CompoundPred.model_pretrained(path_dir = f'/work/09735/yichao/ls6/zhilian/clm_code/tdc_val/ckpts/{minor}_model_{MODEL_EPOCH}')
    # do sample
    raw_data_path = f'/work/09735/yichao/ls6/zhilian/clm_code/admet_{main}_{minor}_test.csv'
    sample_data(raw_data_path, prefix='', overwrite=OVERWRITE)
    for part_n in range(PART_N):
      data_name = f'admet_{main}_{minor}_test_{part_n}'
      base = f"/work/09735/yichao/ls6/zhilian/clm_code/eval_gen/eval_admet_{main}_{minor}_epoch{EPOCH}_part{part_n}_n{max_num}"
      Path(base).mkdir(exist_ok=True, parents=True)
      gen_opt = {
        "num_samples": max_num,
        "model_choice": "transformer",
        "model_path": "/work/09735/yichao/ls6/zhilian/clm_code/raw_pretrain_frag/checkpoint",
        "vocab_path": "/work/09735/yichao/ls6/zhilian/clm_code",
        "epoch": str(EPOCH),
        "save_directory": base,
        "data_path": "/work/09735/yichao/ls6/zhilian/clm_code/sample_data",
        "test_file_name": data_name,
        "batch_size": 64,
        "decode_type": "multinomial",
      }
      if OVERWRITE:
        gen_opt['overwrite'] = OVERWRITE
      gen_script_list = [DRUG_BIN, '/work/09735/yichao/ls6/zhilian/clm_code/generate.py']
      for key, value in gen_opt.items():
        gen_script_list.append(f'--{key}'.replace('_', '-'))
        gen_script_list.append(str(value))
      # generate molecules
      try:
        subprocess.run(gen_script_list, check=True, text=True, capture_output=True)
      except subprocess.CalledProcessError as e:
        # 打印错误输出
        print("Error occurred:", e.stderr)
        raise e

      # combine molecules
      combine_path = base
      combine_script_list = [DRUG_BIN, '/work/09735/yichao/ls6/zhilian/clm_code/combine_mol.py', '--rootFolder', combine_path]
      if OVERWRITE:
        combine_script_list.append('--overwrite')
        combine_script_list.append('True')
      try:
        subprocess.run(combine_script_list, check=True, text=True, capture_output=True)
      except subprocess.CalledProcessError as e:
        # 打印错误输出
        print("Error occurred:", e.stderr)
        raise e

      # prepare cano similes data
      cal_path = f"{base}/generated_molecules_complete.csv"
      print(f'handling path: {cal_path}')
      df = pd.read_csv(cal_path)
      # seq or bool
      value_type = df['value_type'][0]
      
      # caluate accuracy
      if value_type == 'seq':
        ratios = []
        for num in NUMS:
          seq_total = 0
          dir_acc_total = 0
          spec_acc_total = 0
          # for 每个分子
          for _, row in df.iterrows():
            row_gen_smis = []
            for i in range(1, max_num + 1):
              key = f'Predicted_smi_{i}'
              if not key in df or i > num:
                break
              smi = row[key]
              if isinstance(smi, float) and math.isnan(smi):
                continue
              row_gen_smis.append(smi)
            if len(row_gen_smis) == 0:
              continue
            seq_total = seq_total + len(row_gen_smis)
            gen_data_loader = utils.data_process(X_drug = row_gen_smis, y = [-1] * len(row_gen_smis), drug_encoding = DRUG_ENCODDING, split_method='no_split')
            predict_gen = model.predict(gen_data_loader)
            delta_value = row['Delta_Value']
            dir_acc_num, spec_acc_num = stat_seq(row['cpd1Value'], predict_gen, delta_value)
            dir_acc_total = dir_acc_total + dir_acc_num
            spec_acc_total = spec_acc_total + spec_acc_num

          dir_acc_ratio = dir_acc_total / seq_total
          spec_acc_ratio = spec_acc_total / seq_total
          ratios.append((spec_acc_ratio, dir_acc_ratio))
        
        temp_df = pd.DataFrame(columns=['sample_num', 'dir_acc', 'spec_acc'])
        for sample_num, seq_ratios in list(zip(NUMS, ratios)):
          spec_acc_ratio, dir_acc_ratio = seq_ratios
          temp_df = pd.concat([temp_df, pd.DataFrame({'sample_num': [sample_num], 'dir_acc': [dir_acc_ratio], 'spec_acc': [spec_acc_ratio]})], ignore_index=True)
        table = wandb.Table(dataframe=temp_df)
        artifact.add(table, f'{main}_{minor}_seq_part{part_n}')
        # draw
        plt_temp = draw_seq(NUMS, [r[0] for r in ratios], [r[1] for r in ratios])
        wandb.log({f"{main}_{minor}_seq_part{part_n}": plt_temp})
      elif value_type == 'bool':
        ratios = []
        for num in NUMS:
          bool_total = 1
          acc_total = 0
          # for 每个分子
          for _, row in df.iterrows():
            row_gen_smis = []
            for i in range(1, max_num + 1):
              key = f'Predicted_smi_{i}'
              if not key in df or i > num:
                break
              smi = row[key]
              if isinstance(smi, float) and math.isnan(smi):
                continue
              row_gen_smis.append(smi)

            gen_data_loader = utils.data_process(X_drug = row_gen_smis, y = [-1] * len(row_gen_smis), drug_encoding = DRUG_ENCODDING, split_method='no_split')
            predict_gen = model.predict(gen_data_loader)
            delta_value = row['Delta_Value']

            acc_num = stat_bool(row['cpd1Value'], predict_gen, delta_value)
            acc_total = acc_total + acc_num

          acc_ratio = acc_total / bool_total
          ratios.append((acc_ratio))

          temp_df = pd.DataFrame(columns=['sample_num', 'acc'])
          for sample_num, ratio in list(zip(NUMS, ratios)):
            temp_df = pd.concat([temp_df, pd.DataFrame({'sample_num': [sample_num], 'acc': [ratio] })], ignore_index=True)
          table = wandb.Table(dataframe=temp_df)
          artifact.add(table, f'{main}_{minor}_bool_part{part_n}')
          # draw
          plt_temp = draw_bool(NUMS, ratios)
          wandb.log({f"{main}_{minor}_bool_part{part_n}": plt_temp})

# 结束时间
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The evaluation process took {elapsed_time} seconds to complete.")
wandb.log_artifact(artifact)
wandb.finish()