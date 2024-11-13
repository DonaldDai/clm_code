from my_toolset.my_utils import canonic_smiles
import pandas as pd
import numpy as np
from generate import GenerateRunner
from pathlib import Path
import matplotlib.pyplot as plt
from combine_mol import get_completeMol
import argparse
from multiprocessing import Pool
import time
import wandb
from glob import glob


start_time = time.time()
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

# hyper params
EPOCH = 20
PROCESS_N = 24
PART_N = 5
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
    notes=f"eval reproducibility {formatted_time}| EPOCH {EPOCH}",
    tags=["eval", "reproducibility", f"epoch{EPOCH}"],
    name=f"eval reproducibility {formatted_time}| EPOCH {EPOCH}"
)
artifact = wandb.Artifact('reproducibility_table', type='dataset')

def draw(x, y):
  plt.figure(figsize=(10, 5))
  plt.plot(x, y, marker='o', linestyle='-', color='lightcoral', markerfacecolor='peachpuff')
  plt.title('Reproduction Rates by Sample Count')
  plt.xlabel('Sample Count')
  plt.ylabel('Reproduction Rate')
  plt.grid(True, linestyle='--', color='grey', alpha=0.5)  # 使用更柔和的网格线
  return plt

file_list = []

# for file in glob("/work/09735/yichao/ls6/zhilian/clm_code/sample_data/rm_target_*_filtered.csv"):
#   p = Path(file)
#   file_list.append((p.stem, f"/work/09735/yichao/ls6/zhilian/clm_code/eval_gen/eval_produce_{p.stem}_epoch{EPOCH}_n{max_num}"))
# file_list = [('rm_target_pick_test', f"/work/09735/yichao/ls6/zhilian/clm_code/eval_gen/eval_produce_high_epoch{EPOCH}_n{max_num}")]
for part_n in range(PART_N):
  data_name = f'rm_target_pick_test_{part_n}'
  base = f"/work/09735/yichao/ls6/zhilian/clm_code/eval_gen/eval_produce_epoch{EPOCH}_part{part_n}_n{max_num}"
  file_list.append((data_name, base))
for data_name, base in file_list:
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
    "batch_size": 384,
    "decode_type": "multinomial",
    "overwrite": OVERWRITE,
  }
  gen_args = argparse.Namespace(**gen_opt)
  combine_path = base


  # generate molecules
  runner = GenerateRunner(gen_args)
  runner.generate(gen_args)

  # combine molecules
  get_completeMol(combine_path, overwrite=OVERWRITE)

  # prepare cano similes data
  cal_path = f"{base}/generated_molecules_complete.csv"
  print(f'handling path: {cal_path}')
  df = pd.read_csv(cal_path)
  total = len(df['Target_Mol'])
  print(f'Total: {total}')
  tar_set = set(df['Target_Mol'].apply(canonic_smiles))

  # pre cano
  for i in range(1, max_num + 1):
    key = f'Predicted_smi_{i}'
    df[key] = df[key].apply(canonic_smiles)
  # multi process
  def process_column(column_data):
      key, series = column_data
      return key, series.apply(canonic_smiles)
  with Pool(processes=PROCESS_N) as pool:
    # 创建一个异步结果列表
    tmp_results = []
    
    # 对每列数据调用 apply_async
    for key in df.columns:
        result = pool.apply_async(process_column, args=((key, df[key]),))
        tmp_results.append(result)
    
    # 从结果中获取数据并更新 DataFrame
    for result in tmp_results:
        key, series = result.get()
        df[key] = series
  ratios = []
  # caluate reproducibility
  for num in NUMS:
    gen_set = set()
    for i in range(1, max_num + 1):
      key = f'Predicted_smi_{i}'
      if not key in df or i > num:
        break
      # print(f'===handling {key}')
      gen_set = gen_set.union(df[key])

    ratio = len(tar_set.intersection(gen_set))/total
    print(f'Model(sample {num}) can reproduce {ratio} from test dataset')
    ratios.append(ratio * 100)
  print(f'ratios: {ratios}')
  temp_df = pd.DataFrame(columns=['sample_num', 'reproducibility'])
  for sample_num, reproducibility in list(zip(NUMS, ratios)):
     temp_df = temp_df.append({'sample_num': sample_num, 'reproducibility': reproducibility}, ignore_index=True)
  table = wandb.Table(dataframe=temp_df)
  artifact.add(table, f'reproducibility_{data_name}')
  # draw
  plt_temp = draw(NUMS, ratios)
  wandb.log({f"reproducibility_{data_name}": plt_temp})
  

# 结束时间
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The evaluation process took {elapsed_time} seconds to complete.")
wandb.log_artifact(artifact)
wandb.finish()