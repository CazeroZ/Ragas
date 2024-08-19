#写一个extractor类，接受一段文字，调用chatgpt api提取关键字并返回
import requests
import json
import re
import os
import sys
class Extractor:
    def __init__(self,model='gpt-3.5-turbo',api_key=None):
        self.model= model
        self.headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"}

    def get_keywords(self,string):
            keywords = string[string.find("[")+1:string.find("]")]
            keyword_list = keywords.split(',')
            print(keyword_list)
            return keyword_list
    
    def __call__(self, text):
        messages = [
            {
                "role": "system",
                "content": "You're a medical assistant with extensive knowledge of fundus diseases. You helped me extract keywords related to fundus diseases from the questions and the patient's disease descriptions. Please:\
                    1.  Extract the main keywords that represent the core concepts of the question.\
                    2.  Rank these keywords by their relevance to the topic of eye diseases.\
                    3.  If possible, briefly explain why these keywords were selected and how they relate to the question.\
                    Here are some examples:\
                    Input:\
                    Congenital optic disc pit with retinal detachment in left eye.   Temporal fovea of optic disc, peripheral pigmentation, retinal detachment above optic disc macula.\
                    Output:\
                    Keyword:[congenital optic disc pit,retinal detachment,temporal fovea,pigmentation]\
                    Input:\
                    A 21-year-old male with binocular congenital retinoschisis had a visual acuity of 0.15 on the right and 0.12 on the left.  Color fundus images show that the macular area of both eyes has a honeycomb appearance.\
                    Output:\
                    Keyword:[binocular congenital retinoschisis,macular area]\
                    Input:\
                    Fundus color image of bilateral anpigmented retinitis pigmentosa, the patient has typical clinical manifestations of retinitis pigmentosa (from childhood night blindness, centripetal reduction of visual field). It can be seen that the retina around the vascular arch is bluish gray and there is no osteocyte-like pigment.\
                    Output:\
                    Keyword:[bilateral anpigmented retinitis pigmentosa,childhood night blindness]"
            },
            {
                "role": "user",
                "content": text
            }
        ]
        payloads={
            "model": self.model,
            "messages":messages,
            "max_tokens": 20,
            "top_p": 1
        }
        response = requests.post("https://api.bianxie.ai/v1/chat/completions", headers=self.headers, json=payloads)
        json_data = response.json()
        print(json_data)
        if 'choices' in json_data and json_data['choices']:
            content = json_data['choices'][0]['message']['content']
            content=self.get_keywords(content)
            return content

    

if __name__ == '__main__':
     text = "Are there signs of neovascularization (NYE) adjacent to the veins?"
     api_key = 'REMOVED'
     extractor = Extractor(api_key=api_key)
     keywords = extractor(text)
