import re

# 定义输入和输出文件路径
file1_path = 'result.txt'
file2_path = 'result_withoutRag.txt'
output_file1 = 'sorted_result.txt'
output_file2 = 'sorted_result_withoutRag.txt'

# 函数：读取并解析文件，返回图片路径与对应描述的字典
def parse_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配 "Image path:" 开头的每个图片及其描述，使用正则表达式
    pattern = r'Image path: (.*?)\n(.*?)\n(?=Image path:|\Z)'  # 匹配"Image path: "开头的每一块
    matches = re.findall(pattern, content, re.DOTALL)

    # 将图片路径与描述存储为字典
    image_dict = {match[0]: match[1].strip() for match in matches}
    return image_dict

# 函数：根据图片路径排序字典
def sort_by_image_path(image_dict):
    return dict(sorted(image_dict.items()))

# 函数：将排序后的字典保存到文件
def write_sorted_file(output_path, image_dict):
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_path, description in image_dict.items():
            f.write(f"Image path: {image_path}\n{description}\n\n")

# 读取并解析两个文件
file1_dict = parse_file(file1_path)
file2_dict = parse_file(file2_path)

# 将两个文件按图片路径排序
sorted_file1_dict = sort_by_image_path(file1_dict)
sorted_file2_dict = sort_by_image_path(file2_dict)

# 写入排序后的文件
write_sorted_file(output_file1, sorted_file1_dict)
write_sorted_file(output_file2, sorted_file2_dict)

print("文件已按图片路径排序并保存。")
