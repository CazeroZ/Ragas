import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F

class ImageClassifier:
    def __init__(self, device):
        self.device = device
        self.glaucama_cls = self.load_model(
            '/home/jqxu/Ragas/classifiers/Glaucoma/resnet50_epoch70.pth',
            num_classes=3
        )
        self.diabetic_cls = self.load_model(
            '/home/jqxu/Ragas/classifiers/Diabetic/resnet50_epoch10.pth',
            num_classes=5
        )
        self.transform = self.my_transform((224,224))

    def load_model(self, path, num_classes):
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        model.load_state_dict(torch.load(path))
        model.to(self.device).eval()
        return model

    def my_transform(self, standard_size):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(standard_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_label(self, out, disease):
        out = F.softmax(out, dim=1)
        out = torch.argmax(out, dim=1).item()
        if disease == 'Glaucoma':
            if out == 1:
                return 'Glaucoma'
            elif out == 2:
                return 'Unknown Glaucoma'
        elif disease == 'Diabetic':
            if out != 0:
                return f'Diabetic level {out}'
        return 'Unknown Disease'

    def classify_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        labels = []
        labels.append(self.get_label(self.glaucama_cls(img), 'Glaucoma'))
        labels.append(self.get_label(self.diabetic_cls(img), 'Diabetic'))
        labels = [label for label in labels if label != 'Healthy']
        if not labels:
            labels.append('Healthy')
        return labels


