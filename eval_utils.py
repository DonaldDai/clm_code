import pandas as pd
import numpy as np
from pathlib import Path

# WARNING: 数据采样很费时间
def sample_data(data_path, part_n=5, num=2000, sample_n=1, prefix='reproduce', save_dir='/work/09735/yichao/ls6/zhilian/clm_code/sample_data', overwrite=False):
  p_obj = Path(data_path)
  if not overwrite:
    exist_flag = True
    for idx in range(part_n):
      if not Path(f'{save_dir}/{prefix}{p_obj.stem}_{idx}.csv').exists():
        exist_flag = False
    if exist_flag:
      print('EXIST! SKIP SAMPLE DATA')
      return
  # 按频率提取数据
  # df = pd.read_csv('/home/yichao/zhilian/GenAICode/CLModel_v2_zl/rm_target_cano_test.csv')
  df = pd.read_csv(data_path)
  value_counts = df['fromVarSMILES'].value_counts().sort_values(ascending=False)

  # 将结果平均分成5组
  n = len(value_counts) // part_n
  groups = [value_counts[i:i + n] for i in range(0, len(value_counts), n)]

  # 从每组中随机抽取2000个值，如果组内不足，取全部
  for idx, group in enumerate(groups):
      print(f'group {idx} count: {len(group)}')
      # 从group的index中随机选择，如果不够则取全部
      samples = np.random.choice(group.index, min(num, len(group)), replace=False)
      # 初始化一个空的DataFrame用于存储结果
      sampled_df = pd.DataFrame()
      
      # 遍历每个唯一值并随机抽取一个样本
      for value in samples:
          sample = df[df['fromVarSMILES'] == value].sample(sample_n)  # 随机抽取一个样本
          sample['count'] = value_counts[value]
          sampled_df = pd.concat([sampled_df, sample], ignore_index=True)
      sampled_df.to_csv(f'{save_dir}/{prefix}{p_obj.stem}_{idx}.csv', index=False)
