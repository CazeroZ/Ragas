import torch
from transformers import DPRQuestionEncoder, AutoTokenizer
from PIL import Image
import numpy as np
from datasets import load_from_disk
from KeepFIT.keepfit.modeling.model import KeepFITModel
class TextRetriever:
    def __init__(self, dataset_path, device):
        # 初始化设备
        self.device = device
        
        # 加载DPR模型和Tokenizer
        self.encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        
        # 加载数据集
        self.dataset = load_from_disk(dataset_path)
        self.dataset.add_faiss_index(column="embeddings")

    def retrieve(self, query, k=5):
        # 准备查询
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 移动到正确的设备
        with torch.no_grad():
            query_embeddings = self.encoder(**inputs).pooler_output.cpu().numpy()
        
        # 检索最近的k个文档
        scores, retrieved_examples = self.dataset.get_nearest_examples("embeddings", query_embeddings, k=k)
        return scores, retrieved_examples

    def print_results(self, scores, examples):
        # 打印检索结果
        for score, title, text in zip(scores, examples["title"], examples["text"]):
            print(f"Score: {score:.4f}")
            print(f"Title: {title}")
            print(f"Text: {text}")
            print("-" * 40)  # 打印分隔线以清晰区分每个结果

class MMRetriever:
    def __init__(self,dataset_path,device):
        self.device=device
        self.model = KeepFITModel(vision_type='resnet_v2', out_path='./output', from_checkpoint=True,
                         vision_pretrained=True,
                         weights_path=f'/home/jqxu/Ragas/KeepFIT/results/pretraining/50%flair+MM/KeepFIT (50%flair+MM).pth')
        self.dataset=load_from_disk(dataset_path)
        self.dataset.add_faiss_index(column="features")
    def retrieve(self,image_pth,k=1):
        image=Image.open(image_pth).convert("RGB")
        processed_img=self.model.preprocess_image(np.float32(image))
        processed_img = processed_img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature=self.model.vision_model(processed_img)
            image_feature_np = image_feature.cpu().numpy()  # 将特征转换为 Numpy 数组
        scores,retrieved_examples=self.dataset.get_nearest_examples("features",image_feature_np,k=k)
        return scores, retrieved_examples
    def print_results(self,scores,examples):
        for score, image_path, description in zip(scores, examples["image_path"], examples["description"]):
            print(f"Score: {score:.4f}")
            print(f"Image_path: {image_path}")
            print(f"description: {description}")
            print("-" * 80)  # 打印分隔线以清晰区分每个结果

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset_path="/home/jqxu/Ragas/MM_feature_set"
# ImageRetriever=MMRetriever(dataset_path,device)
# image_pth="/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP/figure01-01A.jpg"
# scores,examples=ImageRetriever.retrieve(image_pth)
# ImageRetriever.print_results(scores,examples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/home/jqxu/Ragas/text_datasets/book1_en"
retriever = TextRetriever(dataset_path, device)

query = "Are there signs of neovascularization (NYE) adjacent to the veins?"
scores, examples = retriever.retrieve(query, k=5)

# 打印结果
retriever.print_results(scores, examples)
