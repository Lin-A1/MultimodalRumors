import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes,text_model):
        super(MultimodalClassifier, self).__init__()
        # 文本编码器（BERT）
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.text_hidden_size = self.text_encoder.config.hidden_size
        
        # 图像编码器（ResNet50）
        self.image_encoder = resnet50(weights=None)
        self.image_hidden_size = 2048
        self.image_encoder.fc = nn.Identity()  # 去掉最后的分类层
        
        # 特征融合层
        self.fusion_layer = nn.Linear(self.text_hidden_size + self.image_hidden_size, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, images):
        # 文本特征提取
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # [batch_size, hidden_size]

        # 图像特征提取
        image_features = self.image_encoder(images)  # [batch_size, 2048]

        # 多模态特征融合
        combined_features = torch.cat((text_features, image_features), dim=1)  # [batch_size, hidden_size + 2048]
        fused_features = self.relu(self.fusion_layer(combined_features))
        fused_features = self.dropout(fused_features)

        # 分类层
        logits = self.classifier(fused_features)
        return logits
