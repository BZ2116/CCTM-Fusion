"""
author: Bruce Zhao
date: 2025/5/21
模型定义文件，改进为 EnhancedFusionModel，支持多架构融合、轻量特征融合和优化
"""

import torch
import torch.nn as nn
import timm
from timm.models import list_models

class SpatialAttention(nn.Module):
    """空间注意力模块，用于增强 CNN 的局部特征"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True).unsqueeze(2)  # [batch_size, 1, 1]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out = max_out.unsqueeze(2)  # [batch_size, 1, 1]
        out = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 2, 1]
        out = self.conv(out).squeeze(2)  # [batch_size, 1]
        return self.sigmoid(out) * x

class MixNorm(nn.Module):
    """混合归一化模块，结合 BatchNorm 和 LayerNorm"""
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        bn_out = self.bn(x)
        ln_out = self.ln(x)
        return self.alpha * bn_out + (1 - self.alpha) * ln_out

class MLPMixer(nn.Module):
    """MLP-Mixer 模块，包含 Token-Mixing 和 Channel-Mixing"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.channel_mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

class CrossModelAttention(nn.Module):
    """混合跨模型注意力模块，结合卷积和多头注意力"""
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.split_dim = input_dim // 2
        self.conv = nn.Conv1d(self.split_dim, self.split_dim, kernel_size=1, groups=4)  # 分组卷积
        self.attention = nn.MultiheadAttention(embed_dim=self.split_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        split_features = torch.split(features, [self.split_dim, self.split_dim], dim=2)
        conv_out = self.conv(split_features[0].transpose(1, 2)).transpose(1, 2)
        attn_out, _ = self.attention(split_features[1].transpose(0, 1), split_features[1].transpose(0, 1), split_features[1].transpose(0, 1))
        attn_out = attn_out.transpose(0, 1)
        combined = torch.cat([conv_out, attn_out], dim=2)
        return self.norm(combined + self.dropout(combined))

class TransformerFusion(nn.Module):
    """Transformer 融合层，带多阶段融合和正则化"""
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 中间融合：逐元素平均
        intermediate = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([x, intermediate], dim=1)
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        weights = torch.softmax(self.weight_layer(x), dim=0)
        x = (x * weights).sum(dim=0)
        return x

class BaselineModel(nn.Module):
    """单模型基线分类器"""
    def __init__(self, model_name, weight_path, num_classes=10, input_size=224):
        super().__init__()
        valid_models = list_models()
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name: {model_name}")

        self.upsample = nn.AdaptiveAvgPool2d((input_size, input_size))
        self.base_model = timm.create_model(model_name, pretrained=False, num_classes=0)
        try:
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True)
            self.base_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise ValueError(f"Error loading weights for {model_name}: {str(e)}")

        feat_dim = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            MixNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.upsample(x)
        feat = self.base_model(x)
        return self.classifier(feat)

class FusionModel(nn.Module):
    """EnhancedFusionModel，支持多架构融合和优化"""
    def __init__(self, base_model_names, weight_paths, input_size=224, num_classes=10,
                 proj_dim=512, cls_hidden_dim=1024, freeze_layers_ratio=0.8, num_heads=8):
        super().__init__()
        if len(base_model_names) != len(weight_paths):
            raise ValueError("Number of model names must match number of weight paths")
        valid_models = list_models()
        for name in base_model_names:
            if name not in valid_models:
                raise ValueError(f"Invalid model name: {name}")

        self.upsample = nn.AdaptiveAvgPool2d((input_size, input_size))
        self.base_models = nn.ModuleList()
        self.projections = nn.ModuleList()

        for name, weight_path in zip(base_model_names, weight_paths):
            model = timm.create_model(name, pretrained=False, num_classes=0)
            try:
                state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise ValueError(f"Error loading weights for {name}: {str(e)}")

            num_layers = len(list(model.parameters()))
            num_freeze = int(num_layers * freeze_layers_ratio)
            for i, param in enumerate(model.parameters()):
                if i < num_freeze:
                    param.requires_grad = False
            self.base_models.append(model)

            feat_dim = model.num_features
            if "resnet" in name.lower():
                proj = nn.Sequential(
                    nn.Linear(feat_dim, proj_dim),
                    SpatialAttention(proj_dim),
                    MixNorm(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            elif "vit" in name.lower():
                proj = nn.Sequential(
                    nn.Linear(feat_dim, proj_dim),
                    MixNorm(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(proj_dim, proj_dim)
                )
            elif "efficientnet" in name.lower():
                proj = nn.Sequential(
                    nn.Linear(feat_dim, proj_dim),
                    MixNorm(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            else:  # MLP-Mixer
                proj = nn.Sequential(
                    MLPMixer(feat_dim, hidden_dim=proj_dim),
                    nn.Linear(feat_dim, proj_dim),
                    MixNorm(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            self.projections.append(proj)

        self.cross_attention = CrossModelAttention(input_dim=proj_dim, num_heads=num_heads, dropout=0.1)
        self.fusion = TransformerFusion(input_dim=proj_dim, num_heads=num_heads, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, cls_hidden_dim),
            MixNorm(cls_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(cls_hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.upsample(x)
        features = []
        for model, proj in zip(self.base_models, self.projections):
            feat = model(x)
            feat = proj(feat)
            features.append(feat)

        combined = torch.stack(features, dim=1)
        combined = self.cross_attention(combined)
        fused = self.fusion(combined)
        return self.classifier(fused)

    def get_optimizer_params(self, base_lr=1e-4, proj_lr=1e-3, cls_lr=1e-3):
        """返回差分学习率的参数组"""
        base_params = []
        proj_params = []
        cls_params = []
        for model in self.base_models:
            base_params.extend([p for p in model.parameters() if p.requires_grad])
        for proj in self.projections:
            proj_params.extend(proj.parameters())
        cls_params.extend(self.cross_attention.parameters())
        cls_params.extend(self.fusion.parameters())
        cls_params.extend(self.classifier.parameters())
        return [
            {'params': base_params, 'lr': base_lr, 'weight_decay': 1e-5},
            {'params': proj_params, 'lr': proj_lr, 'weight_decay': 1e-5},
            {'params': cls_params, 'lr': cls_lr, 'weight_decay': 1e-5}
        ]
