import json
import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv('/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP_translated_v1.csv')

# 构建JSON对象
data = []
for index, row in df.iterrows():
    image_id = row['Image_ID']
    description = row['en_caption']

    # 检查不同类型的图像文件
    for ext in ['png', 'jpeg', 'jpg']:
        image_path = f"/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP/{image_id}.{ext}"
        if os.path.exists(image_path):
            entry = {
                "image_path": image_path,
                "description": description
            }
            data.append(entry)
            break  # 找到一个存在的文件后就跳出循环

# 将JSON对象写入文件
with open('descriptionMM.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)