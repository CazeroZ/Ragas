import json
import markdown
import os

def convert_markdown_to_html(md_text):
    """将Markdown文本转换为HTML"""
    return markdown.markdown(md_text)

def process_json_file(input_json_path, output_json_path):
    """处理JSON文件，转换Markdown为HTML，并保存结果"""
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 遍历每个记录
    for record in data:
        # 确保'result_text'字段存在
        if 'result_text' in record:
            # 将每个Markdown文本转换为HTML
            record['result_text'] = [convert_markdown_to_html(text) for text in record['result_text']]

    # 将处理后的数据保存回新的JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 指定输入JSON文件路径和输出JSON文件路径
    input_json = '/home/jqxu/Ragas/results_tmp/updated_result_Retclip.json'  # 替换为你的输入文件路径
    output_json = '/home/jqxu/Ragas/results_tmp/updated_result_Retclip_mdrm.json'  # 替换为你希望保存的输出文件路径

    process_json_file(input_json, output_json)
