import os
import json

# 统计并打印相似图片组的函数
def find_and_print_similar_images(json_data):
    similar_groups = []
    
    for obj in json_data:
        # 获取图片名称（仅文件名，不包括路径）
        image_name = os.path.basename(obj["image_path"])
        similar_image_name = os.path.basename(obj["similar_image_path"])
        
        # 去掉文件名前缀 "figure" 来进行比较
        image_name_trimmed = image_name[6:]  # 从第6个字符开始（跳过 'figure'）
        similar_image_name_trimmed = similar_image_name[6:]
        
        # 只比较去掉最后一个字符后的前缀
        if image_name_trimmed[:2] == similar_image_name_trimmed[:2] or image_name_trimmed[:-5] == similar_image_name_trimmed[:-5]:
            # 将符合条件的组加入列表
            similar_groups.append((image_name, similar_image_name))
    
    # 打印相似的图片组
    for group in similar_groups:
        print(f"相似图片组: {group[0]} 和 {group[1]}")
    
    # 返回相似图片组数
    return len(similar_groups)

# 示例：假设你已经加载了json文件
with open('/home/jqxu/Ragas/results_tmp/updated_result_Retclip.json', 'r') as f:
    data = json.load(f)

# 获取总的图片组数
total_groups = len(data)

# 计算相似图片组
similar_count = find_and_print_similar_images(data)

# 输出结果
print(f"相似图片组数: {similar_count}")
print(f"总图片组数: {total_groups}")
print(f"相似图片组数/总图片组数 = {similar_count / total_groups:.2f}")
