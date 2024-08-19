from datasets import load_from_disk, concatenate_datasets

# 列出你想要合并的所有数据集路径
dataset_paths = [
   "/home/jqxu/Ragas/text_datasets/dpr-txt/book1_en",
    "/home/jqxu/Ragas/text_datasets/dpr-txt/book2_en",
    "/home/jqxu/Ragas/text_datasets/dpr-txt/book3_en",
    "/home/jqxu/Ragas/text_datasets/dpr-txt/book4_en"
]

# 加载所有数据集
datasets = [load_from_disk(path) for path in dataset_paths]

# 将所有数据集合并为一个
combined_dataset = concatenate_datasets(datasets)

# 验证合并后的数据集
print(combined_dataset)

# 将合并后的数据集保存到磁盘
combined_dataset.save_to_disk("/home/jqxu/Ragas/text_datasets/dpr-txt/combined_books")
