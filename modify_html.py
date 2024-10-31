import json

# 读取JSON文件
with open("./results_tmp/updated_result_Retclip_mdrm.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 创建新的HTML文件
with open("new_result_Retclip.html", "w", encoding="utf-8") as f:
    f.write('<html>\n<head>\n<title>Generated Retinal Report</title>\n')
    f.write('<style>\n')
    f.write('body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }\n')
    f.write('h1, h2, h3, h4 { color: #2c3e50; font-weight: bold; }\n')
    f.write('p { font-size: 16px; color: #333; }\n')  # 设置段落的字体大小为16px
    f.write('.section { margin-bottom: 40px; }\n')
    f.write('.image-container img { width: 400px; border: 1px solid #ddd; padding: 10px; }\n')
    f.write('.report-text { background-color: #f9f9f9; padding: 15px; border: 1px solid #ccc; }\n')
    f.write('.report-questions { margin-top: 20px; font-style: italic; }\n')
    f.write('</style>\n')
    f.write('</head>\n<body>\n')
    f.write('<h1>Retinal Fundus Image Report</h1>\n')

    # 遍历JSON数据
    for index, item in enumerate(data):
        # 替换图片路径前缀为./datasets
        image_path = item["image_path"].replace("/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP/", "./datasets/")
        
        # 写入图片及描述
        f.write(f'<div class="section">\n')
        f.write(f'<h3>{image_path.split("/")[-1]}</h3>\n')
        f.write(f'<div class="image-container">\n')
        f.write(f'<img src="{image_path}" alt="{image_path.split("/")[-1]}">\n')
        f.write('</div>\n')
        
        f.write(f'<p><strong>Original Description:</strong> {item["original_description"]}</p>\n')

        # 写入生成的报告，正确处理Markdown格式为HTML
        f.write('<h3>Generated Report:</h3>\n')
        f.write('<div class="report-text">\n')

        for paragraph in item["result_text"]:
            # 将 Markdown 样式转换为 HTML
            f.write(f'<p>{paragraph}</p>\n')

        f.write('</div>\n')

        # 添加五个问题及选择框
        f.write('<div class="report-questions">\n')
        f.write('<p>以下五个问题均回答是或否即可:</p>\n')

        # 生成唯一的name属性确保每个问题都有唯一的选项组
        f.write(f'<p>Q1: 诊断是否正确（多个诊断不全也算错误）？<br>')
        f.write(f'<input type="radio" name="q1_{index}" value="yes"> 是\n')
        f.write(f'<input type="radio" name="q1_{index}" value="no"> 否</p>\n')

        f.write(f'<p>Q2: 对每个生物标志（共5个）的描述是否有误/模糊？<br>')
        f.write(f'<input type="radio" name="q2_{index}" value="yes"> 是\n')
        f.write(f'<input type="radio" name="q2_{index}" value="no"> 否</p>\n')

        f.write(f'<p>Q3: 对原因的解释是否有误/模糊？<br>')
        f.write(f'<input type="radio" name="q3_{index}" value="yes"> 是\n')
        f.write(f'<input type="radio" name="q3_{index}" value="no"> 否</p>\n')

        f.write(f'<p>Q4: 文本描述方式是否常用？<br>')
        f.write(f'<input type="radio" name="q4_{index}" value="yes"> 是\n')
        f.write(f'<input type="radio" name="q4_{index}" value="no"> 否</p>\n')

        f.write(f'<p>Q5: 文本内容描述是否清晰？<br>')
        f.write(f'<input type="radio" name="q5_{index}" value="yes"> 是\n')
        f.write(f'<input type="radio" name="q5_{index}" value="no"> 否</p>\n')

        f.write('</div>\n')
        f.write('</div>\n')

    # 关闭HTML结构
    f.write('</body>\n</html>\n')

print("HTML文件已成功生成，Markdown风格文本已正确转换为HTML格式，并添加了选择框。")
