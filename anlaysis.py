import os
import json
import shutil
import re

# 将Markdown样式转换为HTML格式的函数
def markdown_to_html(text):
    # 转换标题
    text = re.sub(r'### (.*?)', r'<h3>\1</h3>', text)
    text = re.sub(r'## (.*?)', r'<h2>\1</h2>', text)
    
    # 转换粗体
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # 转换换行符为段落
    text = re.sub(r'\n\n', r'</p><p>', text)
    text = re.sub(r'\n', r' ', text)
    
    # 添加段落开始和结束标签
    text = '<p>' + text + '</p>'
    
    return text

# 读取更新后的JSON文件
updated_json_file_path = './results_tmp/updated_result_Retclip.json'
with open(updated_json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建目标文件夹
datasets_folder = './datasets'
similarset_folder = './similarset'
os.makedirs(datasets_folder, exist_ok=True)
os.makedirs(similarset_folder, exist_ok=True)

# 移动图像到指定目录
for sample in data:
    # 移动原始图像到datasets文件夹
    original_img_path = sample['image_path']
    original_img_name = os.path.basename(original_img_path)
    new_original_img_path = os.path.join(datasets_folder, original_img_name)
    
    if os.path.exists(original_img_path):
        shutil.copy(original_img_path, new_original_img_path)

    # 移动相似图像到similarset文件夹
    similar_img_path = sample['similar_image_path']
    similar_img_name = os.path.basename(similar_img_path)
    new_similar_img_path = os.path.join(similarset_folder, similar_img_name)
    
    if os.path.exists(similar_img_path):
        shutil.copy(similar_img_path, new_similar_img_path)

# 创建HTML文件
html_output_path = './result_Retclip.html'
with open(html_output_path, 'w', encoding='utf-8') as html_file:
    html_file.write('<html>\n<head>\n<title>Retinal Fundus Image Report</title>\n')
    html_file.write('<style>\n')
    html_file.write('body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }\n')
    html_file.write('.image-container { display: flex; justify-content: space-between; }\n')
    html_file.write('.image-container img { width: 45%; border: 1px solid #ddd; padding: 10px; }\n')
    html_file.write('.section { margin-bottom: 40px; }\n')
    html_file.write('h1, h2, h3 { color: #2c3e50; }\n')
    html_file.write('.report-text { background-color: #f9f9f9; padding: 15px; border: 1px solid #ccc; }\n')
    html_file.write('.keywords { font-weight: bold; color: #34495e; }\n')
    html_file.write('</style>\n')
    html_file.write('</head>\n<body>\n')
    html_file.write('<h1>Retinal Fundus Image Report</h1>\n')
    
    for sample in data:
        original_img_name = os.path.basename(sample['image_path'])
        similar_img_name = os.path.basename(sample['similar_image_path'])
        
        html_file.write('<div class="section">\n')
        
        # 使用flexbox并排显示两张图片，并在图片上方显示实际的文件名
        html_file.write('<div class="image-container">\n')
        html_file.write('<div>\n')
        html_file.write(f'<h3>{original_img_name}</h3>\n')  # 显示原始图像文件名
        html_file.write(f'<img src="./datasets/{original_img_name}" alt="{original_img_name}">\n')
        html_file.write('</div>\n')
        
        html_file.write('<div>\n')
        html_file.write(f'<h3>{similar_img_name}</h3>\n')  # 显示相似图像文件名
        html_file.write(f'<img src="./similarset/{similar_img_name}" alt="{similar_img_name}">\n')
        html_file.write('</div>\n')
        html_file.write('</div>\n')
        
        # 原始描述和关键词
        html_file.write(f'<p class="report-text"><strong>Original Description:</strong> {sample["original_description"]}</p>\n')
        html_file.write(f'<p class="keywords"><strong>Extracted Keywords:</strong> {", ".join(sample["extracted_keywords"])}</p>\n')
        
        # 生成的报告（将markdown格式转换为HTML格式）
        html_file.write('<h3>Generated Report:</h3>\n')
        generated_report = "\n".join(sample["result_text"])
        html_file.write('<div class="report-text">\n')
        html_file.write(markdown_to_html(generated_report))
        html_file.write('</div>\n')
        
        # 添加API分析结果
        if 'comparison_analysis' in sample:
            html_file.write('<h3>Comparison Analysis (Generated vs Real Annotation):</h3>\n')
            html_file.write(f'<div class="report-text"><p>{markdown_to_html(sample["comparison_analysis"])}</p></div>\n')
        
        # Combined Texts
        
        html_file.write('</div>\n')
    
    html_file.write('</body>\n</html>')

print(f"HTML文件已成功生成: {html_output_path}")
