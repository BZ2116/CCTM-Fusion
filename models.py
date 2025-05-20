"""
author: Bruce Zhao
date: 2025/5/20
模型定义文件，改进为 CCTM-Fusion，支持多架构融合、轻量特征融合和小规模数据集优化
"""

import torch
import torch.nn as nn
import timm
from timm.models import list_models


class SpatialAttention(nn.Module):
    """
    空间注意力模块，用于增强 CNN 的局部特征
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, in_channels]
        avg_out = torch.mean(x, dim=1, keepdim=True).unsqueeze(1)  # [batch_size, 1, 1]
        max_out, _ = torch.max(x, dim=1, keepdim=True).unsqueeze(1)  # [batch_size, 1, 1]
        out = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 2, 1]
        out = self.conv(out).squeeze(2)  # [batch_size, 1]
        return self.sigmoid(out) * x


class MixNorm(nn.Module):
    """
    混合归一化模块，结合 BatchNorm 和 LayerNorm，适配小规模数据集
    """

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.ln = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重

    def forward(self, x):
        bn_out = self.bn(x)
        ln_out = self.ln(x)
        return self.alpha * bn_out + (1 - self.alpha) * ln_out


class CrossModelAttention(nn.Module):
    """
    跨模型注意力模块，促进模型间特征交互
    """

    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        # features: [batch_size, num_models, input_dim]
        features = features.transpose(0, 1)  # [num_models, batch_size, input_dim]
        attn_output, _ = self.attention(features, features, features)
        features = self.norm(features + self.dropout(attn_output))
        return features.transpose(0, 1)  # [batch_size, num_models, input_dim]


class TransformerFusion(nn.Module):
    """
    Transformer 融合层，带可学习加权输出
    """

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
        self.weight_layer = nn.Linear(input_dim, 1)  # 可学习权重

    def forward(self, x):
        x = x.transpose(0, 1)  # [num_models, batch_size, input_dim]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        weights = torch.softmax(self.weight_layer(x), dim=0)  # [num_models, batch_size, 1]
        x = (x * weights).sum(dim=0)  # [batch_size, input_dim]
        return x


class BaselineModel(nn.Module):
    """
    单模型基线分类器，用于对比实验
    """

    def __init__(self, model_name, weight_path, num_classes=10, input_size=224):
        super().__init__()
        # 校验模型名称
        valid_models = list_models()
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name: {model_name}")

        # 上采样层
        self.upsample = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)

        # 加载基模型
        self.base_model = timm.create_model(model_name, pretrained=False, num_classes=0)
        try:
            state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True)
            self.base_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise ValueError(f"Error loading weights for {model_name} from {weight_path}: {str(e)}")

        # 获取特征维度
        feat_dim = self.base_model.num_features
        # 分类器
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
    """
    CCTM-Fusion 模型，支持多架构融合、轻量特征融合和小规模数据集优化
    """

    def __init__(self, base_model_names, weight_paths, input_size=224, num_classes=10,
                 proj_dim=512, cls_hidden_dim=1024, freeze_layers_ratio=0.5, num_heads=8):
        super().__init__()
        # 校验输入
        if len(base_model_names) != len(weight_paths):
            raise ValueError("Number of model names must match number of weight paths")
        valid_models = list_models()
        for name in base_model_names:
            if name not in valid_models:
                raise ValueError(f"Invalid model name: {name}")

        # 上采样层
        self.upsample = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)

        # 初始化模型和投影层
        self.base_models = nn.ModuleList()
        self.projections = nn.ModuleList()

        for name, weight_path in zip(base_model_names, weight_paths):
            model = timm.create_model(name, pretrained=False, num_classes=0)
            try:
                state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise ValueError(f"Error loading weights for {name} from {weight_path}: {str(e)}")

            # 冻结部分层
            num_layers = len(list(model.parameters()))
            num_freeze = int(num_layers * freeze_layers_ratio)
            for i, param in enumerate(model.parameters()):
                if i < num_freeze:
                    param.requires_grad = False
            self.base_models.append(model)

            # 自适应投影层
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
                    nn.Linear(feat_dim, proj_dim),
                    MixNorm(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            self.projections.append(proj)

        # 跨模型注意力
        self.cross_attention = CrossModelAttention(input_dim=proj_dim, num_heads=num_heads, dropout=0.1)
        # Transformer 融合层
        self.fusion = TransformerFusion(input_dim=proj_dim, num_heads=num_heads, dropout=0.1)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, cls_hidden_dim),
            MixNorm(cls_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(cls_hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.upsample(x)  # [batch_size, channels, input_size, input_size]
        features = []
        for model, proj in zip(self.base_models, self.projections):
            feat = model(x)  # [batch_size, feat_dim]
            feat = proj(feat)  # [batch_size, proj_dim]
            features.append(feat)

        # 跨模型交互
        combined = torch.stack(features, dim=1)  # [batch_size, num_models, proj_dim]
        combined = self.cross_attention(combined)  # [batch_size, num_models, proj_dim]
        fused = self.fusion(combined)  # [batch_size, proj_dim]
        return self.classifier(fused)  # [batch_size, num_classes]

    def get_optimizer_params(self, base_lr=1e-4, proj_lr=1e-3, cls_lr=1e-3):
        """
        返回差分学习率的参数组，包含 L2 正则化
        """
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