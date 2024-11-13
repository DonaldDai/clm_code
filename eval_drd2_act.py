import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import wandb
import subprocess
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import load

np.random.seed(678)

start_time = time.time()
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

# hyper params
EPOCH = 20
DRUG_BIN = f'/work/09735/yichao/ls6/miniconda/envs/drugassist-jay/bin/python'
SEQ_VAL_R = [1, 5, 10, 15]
NUMS = [1]
OVERWRITE = False
for i in range(5, 16, 5):
  NUMS.append(i)
for i in range(20, 501, 20):
  NUMS.append(i)
max_num = max(NUMS)
print(f'USING NUMS: {NUMS}, max num: {max_num}')
run = wandb.init(
    project="CLM",
    notes=f"eval drd2 act {formatted_time}| EPOCH {EPOCH}",
    tags=["eval", "drd2", 'act', f"epoch{EPOCH}"],
    name=f"eval drd2 act {formatted_time}| EPOCH {EPOCH}"
)
artifact = wandb.Artifact('Drd2_ACT_table', type='dataset')

def draw_seq(*args, labels=[]):
  plt.figure(figsize=(10, 5))
  for idx, data in enumerate(args):
    if idx == 0:
      continue
    plt.plot(args[0], data, label=labels[idx-1], marker='o', linestyle='-', markerfacecolor='peachpuff')
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

def smiles_to_features(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def parse_seq_num(delta_value_str):
  n1_str, n2_str = delta_value_str[1:-1].split(',')
  return float(n1_str), float(n2_str)

def stat_seq(originFrom, originTo, pred_y, delta_value, rs=[]):
  dir_acc_num = 0
  spec_acc_num = 0
  n1, n2 = parse_seq_num(delta_value)
  n_left = originFrom + n1
  n_right = originFrom + n2
  for y in pred_y:
    if y > n_left and y <= n_right:
      spec_acc_num = spec_acc_num + 1
    if (n1 < 0 or n2 < 0) and y - originFrom <= 0:
      dir_acc_num = dir_acc_num + 1
    elif (n1 > 0 or n2 > 0) and y - originFrom > 0:
      dir_acc_num = dir_acc_num + 1
  ret = {
    "dir_acc_num": dir_acc_num,
    "spec_acc_num": spec_acc_num,
  }
  for r in rs:
    ret[f"err_{r}"] = (np.abs((pred_y - originTo) / pred_y) <= r/100).sum()
  return ret

raw_df = pd.read_csv('/work/09735/yichao/ls6/zhilian/clm_code/sample_data/drd2_ki_test.csv')
df = raw_df.sample(n=min([2000, len(raw_df)]))
print(f'sample len: {len(df)}')
df.to_csv('/work/09735/yichao/ls6/zhilian/clm_code/sample_data/drd2_ki_test_sample.csv')
data_name = f'drd2_ki_test_sample'
base = f"/work/09735/yichao/ls6/zhilian/clm_code/eval_gen/eval_drd2_act_epoch{EPOCH}_n{max_num}"
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
# 分为分类和回归两个任务评估
for eval_type in ['bool', 'seq']:
  if eval_type == 'seq':
    model = load('/work/09735/yichao/ls6/zhilian/clm_code/drd2_act_svr_model.joblib')
    ratios = []
    all_keys = ['dir_acc_num', "spec_acc_num"]
    all_keys.extend([f'err_{r}' for r in SEQ_VAL_R])
    for num in NUMS:
      seq_total = 0
      dir_acc_total = 0
      spec_acc_total = 0
      total_record = {key: 0 for key in all_keys}
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
        features = np.array([smiles_to_features(sm) for sm in row_gen_smis])
        predict_gen = model.predict(features)
        delta_value = row['Delta_Value']
        # 1. 优化是否在区间内
        # 2. 优化方向是否相同
        # 3. 优化之后与实验值的差距 1 5 10 15 30
        errs = stat_seq(row['cpd1Value'], row['cpd2Value'], predict_gen, delta_value, rs=SEQ_VAL_R)
        for key in all_keys:
          total_record[key] = total_record[key] + errs[key]

      for key in all_keys:
        total_record[key] = total_record[key] / seq_total
      ratios.append(total_record)
    
    columns = ['sample_num']
    columns.extend(all_keys)
    temp_list = []
    for sample_num, seq_ratios in list(zip(NUMS, ratios)):
      temp_list.append({**seq_ratios, 'sample_num': sample_num})
    temp_df = pd.DataFrame(temp_list, columns=columns)
    table = wandb.Table(dataframe=temp_df)
    artifact.add(table, f'drd2_act_svm_seq')
    # draw
    plt_temp = draw_seq(NUMS, *[[r[key] for r in ratios] for key in all_keys], labels=all_keys)
    wandb.log({f"drd2_act_svr_seq": plt_temp})
  elif eval_type == 'bool':
    model = load('/work/09735/yichao/ls6/zhilian/clm_code/drd2_act_svm_model.joblib')
    ratios = []
    for num in NUMS:
      bool_total = 0
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
        bool_total = bool_total + len(row_gen_smis)
        features = np.array([smiles_to_features(sm) for sm in row_gen_smis])
        predict_gen = model.predict(features)

        acc_num = (predict_gen == row['cpd2_act']).sum()
        acc_total = acc_total + acc_num

      acc_ratio = acc_total / bool_total
      ratios.append(acc_ratio)

    temp_df = pd.DataFrame(columns=['sample_num', 'acc'])
    for sample_num, ratio in list(zip(NUMS, ratios)):
      temp_df = pd.concat([temp_df, pd.DataFrame({'sample_num': [sample_num], 'acc': [ratio] })], ignore_index=True)
    table = wandb.Table(dataframe=temp_df)
    artifact.add(table, f'drd2_act_svr_bool')
    # draw
    print(NUMS, ratios)
    plt_temp = draw_bool(NUMS, ratios)
    wandb.log({f"drd2_act_svr_bool": plt_temp})


# 结束时间
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The evaluation process took {elapsed_time} seconds to complete.")
wandb.log_artifact(artifact)
wandb.finish()