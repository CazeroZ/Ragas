import torch
from datasets import load_from_disk
from PIL import Image
from RETCLIP.RET_CLIP.clip.utils import load_from_name  # 删除 tokenize 导入
from tqdm import tqdm

# Step 1: 加载数据集
dataset = load_from_disk("/home/jqxu/Ragas/MM_feature_set/Keepfit")

# Step 2: 加载 CLIP 模型，并强制设置数据类型为 float32
# Step 2: Load the CLIP model and force the data type to float32
model_name = "ViT-B-16"  # Choose the model name
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name(model_name, device=device)

# Ensure the model uses float32
model = model.to(torch.float32)
model.eval()  # Set to evaluation mode




# Step 3: 定义函数来处理每个样本的图像
def generate_image_embeddings(example):
    # 加载图像
    image_path = example["image_path"]
    img = Image.open(image_path)
    
    # 预处理图像并生成图像嵌入
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 强制将图像张量转换为 float32 类型
    
    img_tensor = img_tensor.type(torch.float32)
    
    # 生成图像嵌入
    with torch.no_grad():
        image_embedding = model.encode_image(img=img_tensor) 
    
    # 返回处理后的样本，只保留图像嵌入
    return {
        "features": image_embedding.cpu().numpy().tolist()  # 将 tensor 转换为 list 并返回
    }

# Step 4: 删除 features 列并重新生成图像嵌入
if "features" in dataset.column_names:
    dataset = dataset.remove_columns("features")

# 使用 map 函数为每个样本生成图像嵌入
dataset = dataset.map(generate_image_embeddings, batched=False)
dataset.add_faiss_index(column="features")
# Step 5: 保存带有新图像嵌入的数据集
dataset.save_to_disk("/home/jqxu/Ragas/MM_feature_set/RET-clip")
