import pandas as pd
import json

# 加载CSV文件
file_path = 'skill_builder_data_corrected.csv'  # 替换为实际文件路径
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 1. 只保留需要的列
columns_to_keep = ['user_id', 'problem_id', 'skill_id', 'correct', 'ms_first_response', 'order_id']
df = df[columns_to_keep]

# 2. 删除存在空值的行
df.dropna(inplace=True)

# 3. 重命名列名 ms_first_response 为 timestamp
df.rename(columns={'ms_first_response': 'timestamp'}, inplace=True)

# 4. 映射 problem_id 列为从 0 开始的连续ID，并保存映射关系
problem_id_mapping = {val: idx for idx, val in enumerate(df['problem_id'].unique())}
df['problem_id'] = df['problem_id'].map(problem_id_mapping)
# 5. 映射 skill_id 列为从 0 开始的连续ID，并保存映射关系
skill_id_mapping = {val: idx for idx, val in enumerate(df['skill_id'].unique())}
df['skill_id'] = df['skill_id'].map(skill_id_mapping)
# 6. 保存 problem_id 和 skill_id 的对应关系
problem_skill_mapping = df[['problem_id', 'skill_id']].drop_duplicates().set_index('problem_id')['skill_id'].to_dict()

problem_id_mapping = {int(k): int(v) for k, v in problem_id_mapping.items()}
skill_id_mapping = {int(k): int(v) for k, v in skill_id_mapping.items()}
problem_skill_mapping = {int(k): int(v) for k, v in problem_skill_mapping.items()}

with open('problem_id_mapping.json', 'w') as f:
    json.dump(problem_id_mapping, f, indent=4)
with open('skill_id_mapping.json', 'w') as f:
    json.dump(skill_id_mapping, f, indent=4)
with open('problem2skill.json', 'w') as f:
    json.dump(problem_skill_mapping, f, indent=4)

df.to_csv('interaction.csv', index=False)
# 7. 按 user_id 分组并排序
# df.sort_values(by=['user_id','timestamp'], inplace=True)
print(len(df))
example = 0
# 8. 按用户分组后每50行划分为一组
grouped_data = []
data = []
group_size = 50
for userid, group in df.groupby('user_id'):
    if len(group) < 5:
        continue
    group.sort_values(by=['order_id'], inplace=True)
    num_subgroups = len(group) // group_size  # 计算子组数量
    for i in range(num_subgroups):
        sub_group = group.iloc[i * group_size:(i + 1) * group_size]
        if len(sub_group) < 5:
            continue
        skill_seq = sub_group['skill_id'].tolist()
        problem_seq = sub_group['problem_id'].tolist()
        timestamp_seq = sub_group['timestamp'].tolist()
        correct_seq = sub_group['correct'].tolist()
        if len(skill_seq) < 5:
            break
        grouped_data.append({
            'user_id': example,
            'skill_seq': skill_seq,
            'problem_seq': problem_seq,
            'timestamp_seq': timestamp_seq,
            'correct_seq': correct_seq
        })

        data.append({
            'user_id': example,
            'skill_seq': skill_seq,
            'problem_seq': problem_seq,
            'timestamp': timestamp_seq,
            'correct_seq': correct_seq
        })
        example += 1
# 9. 转换为新的DataFrame
processed_df = pd.DataFrame(grouped_data)
from random import shuffle
import random

random.seed(2026)
shuffle(data)
# 10. 保存结果到CSV文件
output_file_path = 'all_interactions.csv'
processed_df.to_csv(output_file_path, index=False)

print(f"数据处理完成，结果保存为 {output_file_path}")
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1
train_num = int(len(data) * train_ratio)
val_num = int(len(data) * val_ratio)
test_num = len(data) - train_ratio - val_num
train_data = data[:train_num]
val_data = data[train_num:train_num + val_num]
test_data = data[train_num + val_num:]
train_data = pd.DataFrame(train_data)
val_data = pd.DataFrame(val_data)
test_data = pd.DataFrame(test_data)
train_data.to_csv(r'/home/Q23301264./LPDG/dataset/Assistment/TrainSet/train_data.csv', index=False)
val_data.to_csv(r'/home/Q23301264./LPDG/dataset/Assistment/ValSet/val_data.csv', index=False)
test_data.to_csv(r'/home/Q23301264./LPDG/dataset/Assistment/TestSet/test_data.csv', index=False)
with open('datasetinformation.txt', 'w') as f:
    f.write('problem_num:' + str(len(problem_id_mapping)) + '\n')
    f.write('skill_num:' + str(len(skill_id_mapping)) + '\n')
    f.write('example_num:' + str(example) + '\n')
