import json
import requests
from tqdm import tqdm
from get_api_key import read_api_key
class QueryProcessor:
    def __init__(self, headers):
        self.headers = headers

    def process_query(self, generated_report, real_annotation):
        # 定义API请求的内容
        messages = [
            {"role": "system", "content": "You are an expert. Please provide a **concise comparison** between the generated report and the real annotation. The analysis should be no more than 6 sentences, highlighting only the most important differences."},
            {"role": "user", "content": f"Generated report:\n{generated_report}\n\nReal annotation:\n{real_annotation}"}
        ]
        
        # 调用API接口
        response = requests.post("https://api.bianxie.ai/v1/chat/completions", 
                                headers=self.headers, 
                                json={"model": 'gpt-3.5-turbo', "messages": messages, "max_tokens": 3000})
        
        result = response.json()
        return result['choices'][0]['message']['content']

# 读取JSON文件
json_file_path = '/home/jqxu/Ragas/results_tmp/updated_result_Retclip.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


api_key =read_api_key("api_key.txt")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
processor = QueryProcessor(headers)

# 使用tqdm显示进度条
print("开始调用API进行对比分析...")
for sample in tqdm(data, desc="Processing Samples"):
    generated_report = "\n".join(sample["result_text"])
    real_annotation = sample.get("original_description")

    if real_annotation:
        # 调用API获取分析
        comparison_result = processor.process_query(generated_report, real_annotation)
        # 将对比分析结果添加到JSON中
        sample['comparison_analysis'] = comparison_result

# 保存处理后的JSON文件
updated_json_file_path = '/home/jqxu/Ragas/results_tmp/updated_result_Retclip.json'
with open(updated_json_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"API对比分析已完成，结果已保存到新的JSON文件: {updated_json_file_path}")
