import os
import json
import torch
import pandas as pd
from retriever import MMRetriever
import datasets
from datasets import load_from_disk

# 加载数据集
mmdataset = datasets.load_from_disk("/home/jqxu/Ragas/MM_feature_set/RET-clip_modified")
image_dir = "/home/jqxu/Ragas/datasets"

# 获取所有图片路径
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
# 修改图片路径前缀
image_paths = [path.replace("/home/jqxu/Ragas/datasets", "/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP") for path in image_paths]
# 检查是否使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化检索器
mmRetriever = MMRetriever(device=device, dataset_path="/home/jqxu/Ragas/MM_feature_set/RET-clip_modified")

# 用于存储结果的列表
results = []

# 对每张图片进行检索
for img_path in image_paths:
    # 使用检索器找到最相近的图片
    scores, retrieved_examples = mmRetriever.retrieve(img_path, k=3)
    
    # 构建结果字典（不再包含 description）
    result = {
        "original_image": os.path.basename(img_path)
    }

    # 存储每个检索到的图片及其分数，并将 float32 转换为 float
    for i in range(len(retrieved_examples["image_path"])):
        result[f"retrieved_image{i+1}"] = os.path.basename(retrieved_examples["image_path"][i])
        result[f"score{i+1}"] = float(scores[i])  # 将 float32 转为 float

    # 将结果添加到结果列表
    results.append(result)

# 将结果转换为pandas DataFrame
df = pd.DataFrame(results)

# 保存DataFrame为CSV文件
df.to_csv("/home/jqxu/Ragas/results.csv", index=False)

# 也可以将结果保存为JSON文件
with open("/home/jqxu/Ragas/results.json", "w") as f:
    json.dump(results, f, indent=4)
