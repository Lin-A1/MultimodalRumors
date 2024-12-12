import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50,ResNet50_Weights
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q1, k2) / self.k_dim**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        
        return output.squeeze(1)



class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.unsqueeze(1)
        return context_layer
                        

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes,text_model):
        super(MultimodalClassifier, self).__init__()
        # 注意力机制
        self.sa = SelfAttention(num_attention_heads=2, input_size=768, hidden_size=768)
        self.ca = CrossAttention(in_dim1=768, in_dim2=768, k_dim=64, v_dim=64, num_heads=8)
        
        # 文本编码器（xlnet）
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.text_hidden_size = self.text_encoder.config.hidden_size

        # 图像编码器（ResNet50）
        self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()
        self.image_fc = nn.Linear(2048, 768)
        
        # 特征融合层
        self.fusion_layer = nn.Linear(2304, 768)
        self.classifier = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=1) 
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, ocrinput_ids, ocrattention_mask, images):
        # 文本特征提取
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] #[batch_size, 768]
        text_features = self.sa(text_features) # [batch_size, 768]
        text_features = text_features.squeeze(1).squeeze(1)


        # 图像特征提取
        image_features = self.image_encoder(images)  # [batch_size, 1000]
        image_features = self.image_fc(image_features)

        
        # ocr特征提取
        ocr_outputs = self.text_encoder(input_ids=ocrinput_ids, attention_mask=ocrattention_mask)
        ocr_features = ocr_outputs.last_hidden_state[:, 0, :] # [batch_size, 768]
        ocr_features = self.sa(ocr_features) # [batch_size, 768]
        ocr_features = ocr_features.squeeze(1).squeeze(1)

        
        # 多模态特征融合
        ImageAndOcr = self.ca(ocr_features, image_features)
        
        combined_features = torch.cat((text_features, ImageAndOcr, ocr_features), dim=1)  # [batch_size, 768*3]
        combined_features = self.fusion_layer(combined_features)
        combined_features = self.dropout(combined_features)
        
        # 分类层
        logits = self.classifier(combined_features)
        return logits
