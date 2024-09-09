import datasets
import os
mmRetinal = datasets.load_from_disk("/home/jqxu/Ragas/MM_feature_set")
def get_image_description(image_path):
    image_description = mmRetinal.filter(lambda example: example['image_path'] == image_path)
    print(f"Filtering result for {image_path}: {image_description}")
    if len(image_description) > 0:
        description_dict = image_description.to_dict()
        if 'description' in description_dict:
            return description_dict['description'][0] 
        else:
            return "Description field not found"
    else:
        return "Description not found"

image_folder = "/home/jqxu/Ragas/datasets"

image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
full_image_paths = [os.path.join("/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP", os.path.basename(path)) for path in image_paths]
full_image_paths.sort()
output_file = "ground_description.txt"
with open(output_file, "w") as file:
    for image_path in full_image_paths:
        description = get_image_description(image_path)
        print(f"Writing description for {image_path}: {description}")  # 调试输出
        file.write(f"Path: {image_path}\nDescription: {description}\n-------\n")

print(f"Descriptions saved to {output_file}")
