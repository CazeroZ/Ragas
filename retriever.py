
import torch
from transformers import DPRQuestionEncoder, AutoTokenizer
from PIL import Image
import numpy as np
from datasets import load_from_disk
from KeepFIT.keepfit.modeling.model import KeepFITModel
from transformers import PreTrainedTokenizerFast, PreTrainedModel
#from sentence_transformers import SentenceTransformer, models 
from RETCLIP.RET_CLIP.clip.utils import load_from_name  # 删除 tokenize 导入
from tqdm import tqdm
class TextRetriever:
    def __init__(self, dataset_path, device):
        # 初始化设备
        self.device = device
        #self.model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
        """
        self.model= KeepFITModel(vision_type='resnet_v2', out_path='./output', from_checkpoint=True,
                         vision_pretrained=True,
                         weights_path=f'/home/jqxu/Ragas/weights/KeepFIT (50%flair+MM).pth')
        """
        
        
        # 加载DPR模型和Tokenizer
        self.encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        # 加载数据集
        self.dataset = load_from_disk(dataset_path)
        self.dataset.add_faiss_index(column="embeddings")
    def retrieve(self, query, k=2, is_keywords=False):
        if is_keywords:
            # 如果查询是关键词列表，将每个关键词单独搜索
            all_scores = []
            all_examples = {"title": [], "text": [], "embeddings": []}
            for keyword in query:
                inputs = self.tokenizer(keyword, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    query_embeddings = self.encoder(**inputs).pooler_output.cpu().numpy()
                
                scores, retrieved_examples = self.dataset.get_nearest_examples("embeddings", query_embeddings, k=k)
                
                all_scores.extend(scores)
                all_examples["title"].extend(retrieved_examples["title"])
                all_examples["text"].extend(retrieved_examples["text"])
                all_examples["embeddings"].extend(retrieved_examples["embeddings"])
            
            return all_scores, all_examples
        else:
            # 否则假设查询是自然语言字符串
            query_text = query

            # 准备查询
            inputs = self.tokenizer(query_text, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
    def __init__(self, dataset_path, device, model_name="RET_clip"):
        self.model_name = model_name
        if model_name == "RET_clip":
            model_name = "ViT-B-16"  # Choose the model name
            self.model, self.preprocess = load_from_name(model_name, device=device)
            print("loaded ret-clip vit")
        else:
            self.model = KeepFITModel(vision_type='resnet_v2', out_path='./output', from_checkpoint=True,
                                      vision_pretrained=True,
                                      weights_path=f'/home/jqxu/Ragas/KeepFIT/results/pretraining/50%flair+MM/KeepFIT (50%flair+MM).pth')
            self.preprocess = self.model.preprocess_image
            
        # Ensure the model uses float32 and set it to evaluation mode
        self.model = self.model.to(torch.float32)
        self.model.eval()  
        
        # Load dataset and add FAISS index
        self.dataset = load_from_disk(dataset_path)
        self.dataset.add_faiss_index(column="features")
        
        self.device = device
       
    def retrieve(self, image_pth, k=1):
        # Open the image and apply preprocessing
               
        
        image = Image.open(image_pth).convert("RGB")
        processed_img = self.preprocess(image).unsqueeze(0).to(self.device)  # Preprocess the image (PIL -> tensor)
        
        # Obtain the image features from the model
        with torch.no_grad():
            if self.model_name == "RET_clip":
                image_feature = self.model.encode_image(processed_img.type(torch.float32))  # Ensure the image tensor is float32
            else:
                image_feature = self.model.vision_model(processed_img)
            image_feature_np = image_feature.cpu().numpy()  # Convert to numpy array
        
        # Retrieve k+1 nearest samples, ensuring we include similar images but not the input image itself
        scores, retrieved_examples = self.dataset.get_nearest_examples("features", image_feature_np, k=k+1)
        
        # Filter out the original image from the results
        filtered_scores = []
        filtered_examples = {"image_path": [], "description": [], "features": []}
        
        for score, example_path in zip(scores, retrieved_examples["image_path"]):
            if example_path != image_pth:  # Filter out the original image path
                filtered_scores.append(score)
                filtered_examples["image_path"].append(example_path)
                filtered_examples["description"].append(retrieved_examples["description"][retrieved_examples["image_path"].index(example_path)])
                filtered_examples["features"].append(retrieved_examples["features"][retrieved_examples["image_path"].index(example_path)])
                
            if len(filtered_scores) == k:  # Once we have enough results, break the loop
                break
    
        return filtered_scores, filtered_examples

    def print_results(self, scores, examples):
        for score, image_path, description in zip(scores, examples["image_path"], examples["description"]):
            print(f"Score: {score:.4f}")
            print(f"Image_path: {image_path}")
            print(f"description: {description}")
            print("-" * 80)  # Print separator for clarity
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path="/home/jqxu/Ragas/MM_feature_set/RET-clip_modified"
ImageRetriever=MMRetriever(dataset_path,device)
image_pth="/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP/figure04-74A.jpg"
scores,examples=ImageRetriever.retrieve(image_pth)
ImageRetriever.print_results(scores,examples)
'''
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/home/jqxu/Ragas/text_datasets/keepfit/book1_en"
retriever = TextRetriever(dataset_path, device)

query = "retinal detachment"
print(f"Query: {query}")
scores, examples = retriever.retrieve(query, k=5)

# 打印结果
retriever.print_results(scores, examples)
'''
