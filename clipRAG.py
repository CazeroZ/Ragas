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
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import base64
import KeepFIT.keepfit.modeling.model as keepfit 
from torchvision import models
from classifiers.train import my_transform

from datasets import load_from_disk, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer, RagRetriever, DPRQuestionEncoder, DPRContextEncoder, RagTokenizer, RagSequenceForGeneration
from tqdm import tqdm
from faiss import IndexFlatL2

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        query = [item['query'] for item in data]
        img_path = [item['image_path'] for item in data]
    return query,img_path

def get_image_paths(directory: str, number: int = None) -> List[str]:
    image_paths = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):
            image_paths.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return [image_paths[-1]]
            count += 1
    return image_paths

def get_features_from_image_path(model,image_paths):
    images=[(Image.open(image_path).convert("RGB"))for image_path in image_paths]
    images=[torch.tensor(model.preprocess_image(np.array(image)) for image in images)]
    images=torch.stack(images)
    with torch.no_grad():
        image_features=model.vision_model(images)
    return image_features

def find_entry(data, key, value):
    for entry in data:
        if entry.get(key) == value:
            return entry
    return None

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

def image_query(query, image_path, similar_path=None, description=None,headers=None,labels=None,infor=None):
    messages = []
    if similar_path!=None:
        similar_url=git_image.get_file_url(os.path.basename(similar_path))
        reference_message = {
        "role": "system",
        "content":[
                {
                    "type": "text",
                    "text": "Answer Users question with the background information. The frist iamge is the reference image which is similar to query image."
                }
        ]
        }
        background_message={
        "role": "user",
        "content":[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": similar_url
                    }
                },
                {
                    "type": "text",
                    "text": f"The background information you may use:\nThe description of the image below is: {description}"
                }
        ]
        }
        messages.append(reference_message)
        messages.append(background_message)
    # 构造带有目标图片和问题的消息
    image_url=git_image.add_file_to_repo(os.path.basename(image_path))
    for i,label in enumerate(labels):
        messages.append(target_message = {
            "role": "user",
            "content":[
                    {
                        "type": "text",
                        "text": f"the image's labels are {label}.\n \
                                Some information about the labels is: {infor[i]}.\n \
                                Please answer the following question about the image:{query}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }

            ]
        })
    messages.append(target_message = {
            "role": "user",
            "content":[
                    {
                        "type": "text",
                        "text": f"Please answer the following question about the image:{query}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }

            ]
        })
    payloads={
        "model": 'gpt-4o',
        "messages":messages,
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payloads)
    json_data = response.json()
    print(json_data)
    git_image.delete_file_from_repo(os.path.basename(image_path))
    if 'choices' in json_data and json_data['choices']:
        content = json_data['choices'][0]['message']['content']
        return content

def getlabel(out,disease):
    out=F.softmax(out,dim=1)
    if disease=='Glaucoma':
        if out == 1:
            return 'Glaucoma'
        elif out == 2:
            return 'Unknown Glaucoma'
    if disease=='Diabetic':
        if out != 0 :
            return  f'Diabetic level {out}'

def main():
    api_key="sk-t502-PgPqzhUSpTAYsjVSTA94T3BlbkFJHXpMvF0k0Ge7bQiZqDmI"
    device = "cuda"
    text_dataset_path = "/home/jqxu/Ragas/TextRetrive/TextDataset"
    text_index_path = "/home/jqxu/Ragas/TextRetrive/index.faiss"
    json_path="/home/jqxu/Ragas/query.json"
    model = keepfit.KeepFITModel(vision_type='resnet_v2', out_path='./output', from_checkpoint=True, vision_pretrained=True,
                        weights_path=f'Ragas/weights/KeepFIT (50%flair+MM).pth')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    text_dataset = load_from_disk(text_dataset_path)

    # 初始化RAG Retriever和tokenizer
    rag_retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-base",
        index_name="custom",
        passages_path=text_dataset_path,
        index_path=text_index_path,
        device=device,
    )
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

    # 初始化DPR Question Encoder和tokenizer
    dpr_question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
    dpr_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    #glaucama classifier
    glaucama_cls= models.resnet50()
    num_features = glaucama_cls.fc.in_features
    glaucama_cls.fc = nn.Linear(num_features, 3)  
    glaucama_cls.load_state_dict(torch.load('/home/jqxu/Ragas/classifiers/Glaucoma/resnet50_epoch70.pth'))
    glaucama_cls.to(device).eval()
    transform=my_transform((224,224))

    #diabetic classifier
    diabetic_cls = models.resnet50()
    num_features = diabetic_cls.fc.in_features
    diabetic_cls.fc = nn.Linear(num_features, 5)
    diabetic_cls.load_state_dict(torch.load('/home/jqxu/Ragas/classifiers/Diabetic/resnet50_epoch10.pth'))
    diabetic_cls.to(device).eval()

    #encode dataset
    direc = '/home/jqxu/Ragas/datasets'
    image_paths = get_image_paths(direc)
    image_features = get_features_from_image_path(model,image_paths)
    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(np.array(image_features.cpu()))
    data = []
    query_text,query_image_path=read_json(json_path)
    for image_path,user_query in zip(query_image_path,query_text):
        #encodee query image and get labels
        image_path = '/home/jqxu/Ragas/datasets/figure01-01A.jpg'
        img=transform(Image.open(image_path))
        img=img.unsqueeze(0).to(device)
        labels=[]
        labels.append(getlabel(glaucama_cls(img), 'Glaucoma'))
        labels.append(getlabel(diabetic_cls(img), 'Diabetic'))
        if len(labels) ==0:
            labels.append('Healthy')
        image_search_embedding = get_features_from_image_path(model,[image_path])

        #load VQA dataset
        with open('/home/jqxu/Ragas/description.json', 'r') as file:
            file = json.load(file)
            for item in file:
                data.append(item)

        #text RAG:
        with torch.no_grad():
            doc_dicts=[]
            for label in labels:
                input = dpr_tokenizer(label, return_tensors="pt").to("cuda")
                question_hidden_states = dpr_question_encoder(**input).pooler_output
                question_hidden_states = question_hidden_states.detach().cpu().numpy()
                retrieved_doc_embeds, doc_ids, doc_dict = rag_retriever.retrieve(question_hidden_states, n_docs=7)
                doc_dicts.append(doc_dict)

        text_retrive=[]
        for doc_dict in doc_dicts:
            if doc_dict:
                doc_dict = doc_dict[0]  
                if isinstance(doc_dict, dict): 
                    titles = doc_dict.get('title', [])
                    texts = doc_dict.get('text', [])
                    for title, text in zip(titles, texts):
                        text_retrive.append(title + text)
            else:
                print(f"Unexpected type for doc_dict: {type(doc_dict)}")
        

        #find similar image
        distances, indices = index.search(np.array(image_search_embedding.reshape(1, -1).cpu()), 2) #2 signifies the number of topmost similar images to bring back
        distances = distances[0]
        indices = indices[0]
        indices_distances = list(zip(indices, distances))
        indices_distances.sort(key=lambda x: x[1], reverse=True)

        similar_path = get_image_paths(direc, indices_distances[1][0])[0]
        print(similar_path)
        element = find_entry(data, 'image_path', similar_path)

        #generate query
        user_query = 'Who is the person in the image below?'
        prompt = f"""
            Below is a user query, you should answer the query using the background information.

            query:{user_query}
            """

        print(image_query(prompt,image_path,similar_path,element["description"],hearder=headers,labels=labels,infor=text_retrive))

if __name__ == '__main__':
    main()
    #git_image.upload_images_in_directory("/home/jqxu/Ragas/datasets/train1.jpeg")
    #it_image.upload_images_in_directory("/home/jqxu/Ragas/query1.png")
    #url=git_image.add_file_to_repo("train7.jpeg")
    #print(url)
    #git_image.delete_file_from_repo(url)
