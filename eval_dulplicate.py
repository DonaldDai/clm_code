import pandas as pd
from my_toolset.my_utils import canonic_smiles
import time

start_time = time.time()
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
print(f'Start time: {formatted_time}')

PART_N = 5
BASE_DIR = '/work/09735/yichao/ls6/zhilian/clm_code/sample_data'

# 指定.pkl文件的路径
file_path = '/work/09735/yichao/ls6/zhilian/clm_code/train_cano_vc.pkl'
# test_file_list = [f'{BASE_DIR}/admet_distribution_bbb_martins_test_{n}.csv' for n in range(PART_N)]
test_file_list = [f'/work/09735/yichao/ls6/zhilian/clm_code/rm_target_pick_test.csv']

# 使用pandas的read_pickle函数加载.pkl文件
loaded_results = pd.read_pickle(file_path)

for file in test_file_list:
  test_df = pd.read_csv(file)
  print(f"=========Value counts for {file}:")
  print()  # 添加空行以便于阅读输出
  for column_name, reference_counts in loaded_results:
    total = reference_counts.sum()
    value_counts_1 = test_df[column_name].apply(canonic_smiles).value_counts()
    index_intersection = value_counts_1.index.intersection(reference_counts.index)
    intersection_frequencies = reference_counts[index_intersection] / total
    print(f'{column_name} dulplicate: {intersection_frequencies.sum() * 100}%')
    
# 结束时间
end_time = time.time()
# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"The evaluating dulplication process took {elapsed_time} seconds to complete.")