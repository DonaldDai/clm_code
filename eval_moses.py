import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import wandb
import subprocess
import moses

start_time = time.time()
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

# hyper params
EPOCH = 9
DRUG_BIN = f'/work/09735/yichao/ls6/miniconda/envs/drugassist-jay/bin/python'
RANDOM_SEED = 678
OVERWRITE = False
NUMS = [1]
for i in range(5, 16, 5):
  NUMS.append(i)
for i in range(20, 61, 20):
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

def draw(x, y, value_desc='some_value'):
  plt.figure(figsize=(10, 5))
  plt.plot(x, y, marker='o', linestyle='-', color='lightcoral', markerfacecolor='peachpuff')
  plt.title(f'{value_desc} by Sample Count')
  plt.xlabel('Sample Count')
  plt.ylabel(value_desc)
  plt.grid(True, linestyle='--', color='grey', alpha=0.5)  # 使用更柔和的网格线
  return plt

# 抽样数据
data_path = f'/work/09735/yichao/ls6/zhilian/clm_code/sample_data/moses_test.csv'
if OVERWRITE or not Path(data_path).exists():
  raw_df = pd.read_csv('/work/09735/yichao/ls6/zhilian/clm_code/test.csv')
  sampled_df = raw_df.sample(n=1000, random_state=RANDOM_SEED)
  sampled_df.to_csv(data_path, index=False)
  raw_df = None

base = f"/work/09735/yichao/ls6/zhilian/clm_code/eval_gen/eval_moses_epoch{EPOCH}_n{max_num}"
gen_opt = {
  "num_samples": max_num,
  "model_choice": "transformer",
  "model_path": "/work/09735/yichao/ls6/zhilian/clm_code/raw_pretrain_frag/checkpoint",
  "vocab_path": "/work/09735/yichao/ls6/zhilian/clm_code",
  "epoch": str(EPOCH),
  "save_directory": base,
  "data_path": "/work/09735/yichao/ls6/zhilian/clm_code/sample_data",
  "test_file_name": 'moses_test',
  "batch_size": 64,
  "decode_type": "multinomial",
}
if OVERWRITE:
  gen_opt['overwrite'] = OVERWRITE

# generate molecules
gen_script_list = [DRUG_BIN, '/work/09735/yichao/ls6/zhilian/clm_code/generate.py']
for key, value in gen_opt.items():
  gen_script_list.append(f'--{key}'.replace('_', '-'))
  gen_script_list.append(str(value))
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

records = []
df = pd.read_csv(f"{base}/generated_molecules_complete.csv")
# 统计
for num in NUMS:
  print(f'handling {num}')
  smiles = []
  for i in range(1, max_num + 1):
    key = f'Predicted_smi_{i}'
    if not key in df or i > num:
      break
    # 过滤空值
    smiles.extend(df[key].dropna())
  k_value = [1000]
  if len(smiles) >= 10000:
    k_value.append(10000)
  metrics = moses.get_all_metrics(smiles, k=k_value)
  metrics['sample_num'] = num
  records.append(metrics)
# 上传数据
temp_df = pd.DataFrame(records, columns=records[-1].keys())
table = wandb.Table(dataframe=temp_df)
artifact.add(table, f'moses_stat')
# 画图
for key in records[-1].keys():
  if key == 'sample_num':
    continue
  
  temp_plt = draw([r.get('sample_num', 0) for r in records], [r.get(key, 0) for r in records], value_desc=key)
  wandb.log({f"moses_{key}": temp_plt})
# 结束时间
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The evaluation process took {elapsed_time} seconds to complete.")
wandb.log_artifact(artifact)
wandb.finish()