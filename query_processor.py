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

    def process_query(self, image_path: str, description: str = None, labels: List[str] = None, reference_texts: List[str] = None):
        messages = [
            {
                "role": "system",
                "content": "You are an advanced medical assistant specializing in ophthalmology. Your task is to generate a detailed and structured report for a retinal fundus image. The report should include an analysis of the optic disc, macula, retinal blood vessels, retinal background, and cup-to-disc ratio. Use medical terminology accurately and provide reasoning for your observations."
            }
        ]
        
        image_url = self.get_file_url(image_path)
        messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Please generate a detailed report for the following retinal image:"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
        })
        '''
        # 如果有标签信息，并且标签不为 "Unknown Disease"
        if labels and labels[0] != "Unknown Disease":
            
            labels_text = "\n".join([f"Label {i+1}: {label}" for i, label in enumerate(labels)])
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Here are some labels provided for the image:\n{labels_text}"}]
            })
        '''
        # 添加参考文本的信息
        if reference_texts:
            reference_texts_concat = "\n\n".join(reference_texts)
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Reference texts for context:\n{reference_texts_concat}"}]
            })
        
        # 发送请求并生成注释
        print("Sending to OpenAI...")
        response = requests.post("https://api.bianxie.ai/v1/chat/completions", headers=self.headers, json={"model": 'gpt-4o', "messages": messages, "max_tokens": 3000})
        result = response.json()

        # 输出生成的结果
        print(result)
        return result




'''api_key="REMOVED"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
processor = QueryProcessor(headers)
processor.process_query( image_path="/home/jqxu/Ragas/datasets/figure02-17.jpg")`'''
