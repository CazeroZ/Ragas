import json
import os
from PIL import Image
import git_image
import requests
from typing import List

class QueryProcessor:
    def __init__(self, headers):
        self.headers = headers

    def get_file_url(self, file_path):
        """获取文件的 URL。"""
        return git_image.get_file_url(file_path)

    def add_file_to_repo(self, file_name):
        """将文件添加到仓库，并返回 URL。"""
        return git_image.add_file_to_repo(file_name)

    def process_query(self, query: str, image_path: str, similar_path: str = None, description: str = None, labels: List[str] = None, reference_texts: List[str] = None):
        """构造请求并打印消息。"""
        messages = []
        '''
        if similar_path:
            similar_url = self.get_file_url(similar_path)
            messages.extend([
                {"role": "system", "content": [{"type": "text", "text": "Answer Users question with the background information. The first image is the reference image which is similar to query image."}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": similar_url}}, 
                    {"type": "text", "text": f"The background information you may use:\nThe description of the similar image (the first url) is: {description}"}]}
            ])
        '''
        image_url = self.get_file_url(image_path)
        if labels:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Please answer the following question about the image: {query}"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
            labels_text = "\n".join([f"Label {i+1}: {label}" for i, label in enumerate(labels)])
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"All labels information:\n{labels_text}"}]
            })

        # 然后添加所有参考文本的信息
        if reference_texts:
            reference_texts_concat = "\n\n".join(reference_texts)
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Reference texts provided for context:\n{reference_texts_concat}"}]
            })
        
        print("Sending to OpenAI...")
        response = requests.post("https://api.bianxie.ai/v1/chat/completions", headers=self.headers, json={"model": 'gpt-4o', "messages": messages, "max_tokens": 3000})
        print(response.json())


'''
api_key="REMOVED"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
processor = QueryProcessor(headers)
processor.process_query("What is in this image?", "/home/jqxu/Ragas/datasets/train1.jpeg", "/home/jqxu/Ragas/datasets/train2.jpeg", "It's an alien-technology built by the Transformer Optimus Prime!", ["Label1:oven", "Label2: nuclear-powered secret case"], ["Text about oven technology", "Text about secret cases"])
'''