import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import jieba
import logging

logging.getLogger('jieba').setLevel(logging.ERROR)


class MultiModalDataset(Dataset):
    def __init__(self, excel_file, img_dir, tokenizer, max_len=512, transform=None):
        """
        Args:
            excel_file (string): 数据集文件路径 (XLSX文件)
            img_dir (string): 图像文件夹路径
            tokenizer: 用于文本处理的预训练模型的tokenizer
            max_len (int): 文本最大长度
            transform: 图像变换操作（可选）
        """
        self.data = pd.read_excel(excel_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        if 'label' not in self.data.columns:
            self.data['label'] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据项
        item = self.data.iloc[idx]
        text = item['text']
        ocr = item['image_text']
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # 文本处理：tokenization + padding
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0) 
        attention_mask = encoding['attention_mask'].squeeze(0)

        ocrencoding = self.tokenizer(ocr, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        ocrinput_ids = ocrencoding['input_ids'].squeeze(0) 
        ocrattention_mask = ocrencoding['attention_mask'].squeeze(0)

        
        # 图像处理
        images_list = item['images_list'].split('\t') if pd.notna(item['images_list']) else []
        images = []
        
        for img_name in images_list:
            img_path = os.path.join(self.img_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except OSError:
                img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)  
            
            # 应用图像变换（如果有）
            if self.transform:
                img = self.transform(img)
            
            images.append(img)

        # 如果有多个图像，拼接它们
        if len(images) > 1:
            images = torch.cat(images, dim=1)  # 按第二维度拼接多个图像
            # 将图像 resize 成 [3, 224, 224]
            resized_images = F.interpolate(images.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            images = resized_images.squeeze(0)
            
        elif len(images) == 1:
            images = images[0]
        else:
            images = torch.zeros(3, 224, 224)  # 如果没有图像，则返回零图像
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ocrinput_ids': ocrinput_ids,
            'ocrattention_mask': ocrattention_mask,
            'images': images,
            'label': label
        }

