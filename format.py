import json
import re

# 定义解析txt文件并转换为JSON格式的函数
def process_txt_to_json(txt_file_path, json_output_path):
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 以100个'-'字符作为段落分隔符
    sections = content.strip().split('-' * 100)

    parsed_data = []

    # 处理每一个段落
    for section in sections:
        section = section.strip()  # 去除首尾空白
        image_info = {
            "image_path": "",
            "similar_image_path": "",
            "original_description": "",
            "keywords": [],
            "retinal_report": "",
            "combined_texts": ""  # 添加combined texts字段
        }
        
        # 提取 'Image path' 和 'similar image path'
        image_path_match = re.search(r'Image path:\s*(.*)', section)
        if image_path_match:
            image_info['image_path'] = image_path_match.group(1).strip()
        
        similar_image_path_match = re.search(r'similar image path:\s*(.*)', section)
        if similar_image_path_match:
            image_info['similar_image_path'] = similar_image_path_match.group(1).strip()
        
        # 提取 'Original Description'
        original_description_match = re.search(r'Original Description:\s*(.*)', section)
        if original_description_match:
            image_info['original_description'] = original_description_match.group(1).strip()
        
        # 提取 'Extracted Keywords'
        keywords_match = re.search(r'Extracted Keywords:\s*(\[.*\])', section)
        if keywords_match:
            image_info['keywords'] = eval(keywords_match.group(1).strip())
        
        # 提取报告内容：从 'Extracted Keywords' 到 '++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ++++++++++++++++++++++++++++++++++++++++++++'
        report_match = re.search(r'Extracted Keywords:\s*\[.*\](.*?)\+{100}', section, re.S)
        if report_match:
            image_info['retinal_report'] = report_match.group(1).strip()
        
        # 提取 'Combined Texts'
        combined_texts_match = re.search(r'Combined Texts:([\s\S]*)', section)
        if combined_texts_match:
            image_info['combined_texts'] = combined_texts_match.group(1).strip()

        parsed_data.append(image_info)

    # 将结果写入JSON文件
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_data, json_file, indent=4, ensure_ascii=False)

    print(f"JSON文件已成功生成: {json_output_path}")



# 示例使用：读取txt文件并转换为JSON
txt_file_path = "/home/jqxu/Ragas/results_tmp/result_Retclip.txt"  # 替换为你的txt文件路径
json_output_path = "/home/jqxu/Ragas/results_tmp/tmp.json"  # 替换为你想保存的JSON文件路径

# 执行转换函数
process_txt_to_json(txt_file_path, json_output_path)
