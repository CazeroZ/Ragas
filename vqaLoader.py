import pandas as pd
import os

class VQADataLoader:
    def __init__(self, csv_path, image_dir):
        self.csv_path = csv_path
        self.image_dir = image_dir
    
    def retrieve_image_data(self):
        # Load the CSV file
        data = pd.read_csv(self.csv_path)
        
        # List to store the results
        image_data = []
        
        # Iterate over each row in the dataframe
        for index, row in data.iterrows():
            image_id = row['Image_ID']
            # Find the image file in the specified directory
            image_path = self.find_image_path(image_id)
            
            if image_path:
                # Append the dictionary with image path and corresponding details
                image_data.append({
                    'image_path': image_path,
                    'question': row['question'],
                    'gpt_answer': row['gpt_answer'],
                    'correct_answer': row['correct_answer']
                })
        
        return image_data

    def find_image_path(self, image_id):
        extensions = ['.jpeg', '.jpg', '.png', '.tif']
        for file_name in os.listdir(self.image_dir):
            file_base, file_ext = os.path.splitext(file_name)
            if file_base == image_id and file_ext in extensions:
                return os.path.join(self.image_dir, file_name)

    def print_image_data(self, image_data):
        # Print the data in a formatted way
        cnt = 0
        for item in image_data:
            print(f"Image Path: {item['image_path']}")
            print(f"Question: {item['question']}")
            print(f"GPT Answer: {item['gpt_answer']}")
            print(f"Correct Answer: {item['correct_answer']}")
            print("-" * 40)  # Add a separator for readability
        print(f"Total  VQA pairs are: {len(image_data)}")

'''
csv_path = '/home/jqxu/Ragas/VQA.csv'
image_dir = '/mnt/data/public/MM_Retinal_Image/MM_Retinal_dataset_v1/CFP'

retriever = VQADataRetriever(csv_path, image_dir)
image_data = retriever.retrieve_image_data()
retriever.print_image_data(image_data)
'''