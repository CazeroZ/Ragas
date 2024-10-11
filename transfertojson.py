from datasets import load_from_disk
import os

# 加载数据集
dataset_path = "/home/jqxu/Ragas/text_datasets/dpr-txt/combined_books"
dataset = load_from_disk(dataset_path)

# 移除 'embeddings' 列
dataset_no_embeddings = dataset.remove_columns("embeddings")

# 将处理后的数据集保存为 JSON 文件
output_file_path = "/home/jqxu/Ragas/text_datasets/dpr-txt/combined_books_no_embeddings.json"
dataset_no_embeddings.to_json(output_file_path)

print(f"数据集已成功移除 'embeddings' 列，并保存为 {output_file_path}")
