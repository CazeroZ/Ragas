import json
import torch
from tqdm import tqdm
import os
from transformers import PreTrainedTokenizerFast, PreTrainedModel
from datasets import Dataset, Features, Value, Sequence
import multiprocessing
from sentence_transformers import SentenceTransformer, models 
class TextDatasetProcessor:
    def __init__(self,  model: PreTrainedModel, device):
        self.model = model
        self.device = device
        multiprocessing.set_start_method('spawn', force=True)

    def set_json_path(self, json_path):
        self.json_path = json_path

    def load_json(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        titles = []
        texts = []
        for item in tqdm(data, desc="Processing JSON data"):
            combined_title = f"{item['chapter']}  {item['title']}" if item['title'] else item['chapter']
            if 'contents' in item and len(item['contents']) > 0:
                for content in item['contents']:
                    titles.append(combined_title)
                    texts.append(content)
            else:
                print(f"Warning: 'contents' is empty or missing for item with chapter '{item['chapter']}' and title '{item['title']}'")
        return titles, texts
    def embed_texts(self, dataset):
        def embed(batch):
            embeddings = self.model.encode(batch['title'], convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()  
            return {'embeddings': embeddings}

        return dataset.map(
            embed,
            batched=True,
            batch_size=16,  
            features=Features({
                'title': Value('string'),
                'text': Value('string'),
                'embeddings': Sequence(Value('float32'))
            }),
            num_proc=1  
        )
    def process(self, output_dir):
        titles, texts = self.load_json()
        dataset = Dataset.from_dict({
            'title': titles,
            'text': texts
        }, features=Features({
            'title': Value('string'),
            'text': Value('string')
        }))
        dataset = self.embed_texts(dataset)
        
        base_filename = os.path.splitext(os.path.basename(self.json_path))[0]
        save_path = os.path.join(output_dir, base_filename)
        os.makedirs(save_path, exist_ok=True)
        
        dataset.save_to_disk(save_path)
        dataset.add_faiss_index(column='embeddings')
        dataset.get_index('embeddings').save(f'{save_path}/index.faiss')
        print(f"Dataset and index have been saved to {save_path}")

# 使用示例


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
books=["/mnt/data/public/MM_Retinal_Image/texts/13_book1/book1_en.json","/mnt/data/public/MM_Retinal_Image/texts/14_book2/book2_en.json","/mnt/data/public/MM_Retinal_Image/texts/15_book3/book3_en.json","/mnt/data/public/MM_Retinal_Image/texts/16_book4/book4_en.json"]
processor = TextDatasetProcessor(model, device)
for book in books:
    processor.set_json_path(book)   
    processor.process("/home/jqxu/Ragas/text_datasets")
