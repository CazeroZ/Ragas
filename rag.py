# model imports
import faiss
import json
import torch
from openai import OpenAI
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import git_image
import requests
# helper imports
from tqdm import tqdm
import json
import os
import numpy as np
import pickle
from typing import List, Union, Tuple
# visualisation imports
from PIL import Image
import matplotlib.pyplot as plt
import base64
from torch.nn import functional as F
from KeepFIT.keepfit.modeling.model import KeepFITModel
import os
from torchvision import models,transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
api_key="sk-Y8fNy3axmyojM1kcBf7f5a2a66404aF4A94c5b82B13eBe2d"
#load model on device. The device you are running inference/training on is either a CPU or GPU if you have.
device = "cuda"

#model, preprocess = clip.load("ViT-B/32",device=device)
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Base directory for images
base_image_dir = '/mnt/data/jqxu/MM_Retinal_disk/keepfitMM'

def get_image_paths(directory: str, number: int = None) -> List[str]:
    image_paths = []
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            image_paths.append(filename)
            if number is not None and count == number:
                return [image_paths[-1]]
            count += 1
    return image_paths

def get_features_from_image_path(image_paths, model):
    image_features = []
    batch_size = 32
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i + batch_size]
        images = [model.preprocess_image(np.array(Image.open(os.path.join(base_image_dir, image_path)).convert("RGB"))) for image_path in batch_paths]
        images = torch.stack(images)
        with torch.no_grad():
            batch_features = model.vision_model(images)
            image_features.append(batch_features)
        torch.cuda.empty_cache()  # 释放显存
    image_features = torch.cat(image_features, dim=0)
    return image_features

def my_transform(standard_size):
    def _transform(img):
        if img.size == standard_size:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(img)
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(img)
    return _transform

def find_entry(data, key, value):
    for entry in data:
        if entry.get(key).split('/')[-1] == value.split('/')[-1]:  # Only compare filenames
            return entry
    return None

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

def image_query(query, image_path, similar_path=None, description=None, label=None):
    messages = []
    if similar_path is not None:
        similar_url = git_image.get_file_url(os.path.basename(similar_path))
        reference_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Answer Users question with the background information."
                }
            ]
        }
        background_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"The background information you may use:\nThe description of the image below is: {description}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": similar_url
                    }
                },
                {
                    "type": "text",
                    "text": f"The predicted label for this image is: {label}"
                }
            ]
        }
        messages.append(reference_message)
        messages.append(background_message)

    # 构造带有目标图片和问题的消息
    image_url = git_image.add_file_to_repo(os.path.basename(image_path))
    target_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": query
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": f"The predicted label for the query image is: {label}"
            }
        ]
    }
    messages.append(target_message)

    payloads = {
        "model": 'gpt-4o',
        "messages": messages,
        "max_tokens": 300
    }

    # 发送请求
    url = 'https://api.bianxie.ai/v1/chat/completions'
    response = requests.post(url, headers=headers, json=payloads)
    json_data = response.json()
    print(json_data)
    git_image.delete_file_from_repo(os.path.basename(image_path))
    if 'choices' in json_data and json_data['choices']:
        content = json_data['choices'][0]['message']['content']
        return content


def getlabel(out,disease):
    out=F.softmax(out,dim=1)
    out=torch.argmax(out,dim=1).item()
    print(disease,out)
    if disease=='Glaucoma':
        if out == 1:
            return 'Glaucoma'
        elif out == 2:
            return 'Unknown Glaucoma'
    if disease=='Diabetic':
        if out != 0 :
            return  f'Diabetic level {out}'   

def main():
    print("start")
    feature_file = 'image_features.pt' 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    image_paths = get_image_paths(base_image_dir)
    print("Number of images:", len(image_paths))
    model = KeepFITModel(vision_type='resnet_v2', out_path='./output', from_checkpoint=True,
                         vision_pretrained=True,
                         weights_path=f'/home/jqxu/Ragas/KeepFIT/results/pretraining/50%flair+MM/KeepFIT (50%flair+MM).pth')
    model.to(device)
    glaucama_cls= models.resnet50()
    num_features = glaucama_cls.fc.in_features
    glaucama_cls.fc = nn.Linear(num_features, 3)  
    glaucama_cls.load_state_dict(torch.load('/home/jqxu/Ragas/classifiers/Glaucoma/resnet50_epoch70.pth'))
    glaucama_cls.to(device).eval()
    transform=my_transform((224,224))

    diabetic_cls = models.resnet50()
    num_features = diabetic_cls.fc.in_features
    diabetic_cls.fc = nn.Linear(num_features, 5)
    diabetic_cls.load_state_dict(torch.load('/home/jqxu/Ragas/classifiers/Diabetic/resnet50_epoch10.pth'))
    diabetic_cls.to(device).eval()

    image_path = 'datasets/figure1-7.png'

    img=transform(Image.open(image_path))
    img=img.unsqueeze(0).to(device)
    labels=[]
    labels.append(getlabel(glaucama_cls(img), 'Glaucoma'))
    labels.append(getlabel(diabetic_cls(img), 'Diabetic'))
    if len(labels) ==0:
        labels.append('Healthy')

    if os.path.exists(feature_file):
        print("Loading features from file...")
        image_features = torch.load(feature_file)
    else:
        print("Extracting features...")
        image_features = get_features_from_image_path(image_paths, model)
        torch.save(image_features, feature_file)  
        print("Features saved to", feature_file)

   

    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(np.array(image_features.cpu()))
    data = []
    with open('descriptionMM.json', 'r', encoding='utf-8') as file:
        file_data = json.load(file)
        for item in file_data:
            data.append(item)

    image_path = 'figure01-01A.jpg'  # Simplified to only filename


    image_search_embedding = get_features_from_image_path([image_path], model)
    distances, indices = index.search(np.array(image_search_embedding.reshape(1, -1).cpu()),
                                      2)  
    distances = distances[0]
    indices = indices[0]
    indices_distances = list(zip(indices, distances))
    indices_distances.sort(key=lambda x: x[1], reverse=True)

    similar_path = os.path.basename(get_image_paths(base_image_dir, indices_distances[1][0])[0]) 
    print("similar path:",similar_path)
    element = find_entry(data, 'image_path', similar_path)
    user_query="describe the fundus image below"
    prompt = f"""
        Below is a user query, I want you to answer the query using the background information.

        query:{user_query}
        """


    print(image_query(prompt, image_path, similar_path, element["description"]))

if __name__ == '__main__':
    main()
   