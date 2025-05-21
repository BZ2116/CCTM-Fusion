"""
author: Bruce Zhao
date: 2025/5/21
主训练流程文件 - EnhancedFusionModel，适配完整 CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data_loaders
from models import BaselineModel, FusionModel
from train import train_model
from visualize import save_results, plot_training_history, plot_roc_curve
import os

# 设置随机种子
torch.manual_seed(42)

def load_data(dataset_name='cifar10'):
    """加载完整 CIFAR-10 数据集"""
    train_loader, test_loader = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=32,  # 场景 2：batch_size=32
        input_size=224,
        normalize={
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    )
    print(f"Using {len(train_loader.dataset)} training samples and {len(test_loader.dataset)} test samples")
    return train_loader, test_loader

def count_parameters(model):
    """计算模型参数量（单位：百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def train_baselines(train_loader, test_loader, output_dir="results/baselines"):
    """训练基线模型"""
    model_configs = [
        {'name': 'resnet50', 'weight_path': 'weights/resnet50.bin'},
        {'name': 'vit_base_patch16_224', 'weight_path': 'weights/vit_base_patch16_224.bin'},
        {'name': 'efficientnet_b0', 'weight_path': 'weights/efficientnet_b0.bin'},
        {'name': 'mixer_b16_224', 'weight_path': 'weights/mixer_b16_224.bin'}
    ]

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for config in model_configs:
        model_name = config['name']
        weight_path = config['weight_path']
        model_tag = f'Baseline-{model_name}'
        print(f'\n=== Training Baseline: {model_tag} ===')

        model = BaselineModel(model_name=model_name, weight_path=weight_path, num_classes=10)
        param_count = count_parameters(model)
        print(f"Parameters: {param_count:.2f}M")

        # 多 GPU 支持
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        model, history, metrics = train_model(
            model, train_loader, test_loader,
            optimizer, nn.CrossEntropyLoss(),
            model_name=model_tag, num_epochs=20  # 场景 2：20 轮
        )

        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_tag}_weights.pth'))
        plot_roc_curve(model, test_loader, output_dir, model_name=model_tag)

        save_results(model_tag, param_count, metrics, history, output_dir)
        results[model_tag] = (metrics, history)

    return results

def train_fusions(train_loader, test_loader, output_dir="results/fusions"):
    """训练融合模型"""
    model_configs = [
        {'name': 'resnet50', 'weight_path': 'weights/resnet50.bin'},
        {'name': 'vit_base_patch16_224', 'weight_path': 'weights/vit_base_patch16_224.bin'},
        {'name': 'efficientnet_b0', 'weight_path': 'weights/efficientnet_b0.bin'},
        {'name': 'mixer_b16_224', 'weight_path': 'weights/mixer_b16_224.bin'}
    ]

    combinations = [
        [model_configs[0], model_configs[1], model_configs[2], model_configs[3]],  # CCTM-Fusion
    ]

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        model_tag = 'CCTM-Fusion'
        print(f'\n=== Training Fusion: {model_tag} ===')
        model_names = [c['name'] for c in combo]
        weight_paths = [c['weight_path'] for c in combo]
        model = FusionModel(
            model_names, weight_paths, num_classes=10,
            freeze_layers_ratio=0.8, proj_dim=512, cls_hidden_dim=1024, num_heads=8
        )

        param_count = count_parameters(model)
        print(f"Parameters: {param_count:.2f}M")

        # 多 GPU 支持
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        # 阶段 1：冻结基模型
        for param in model.base_models.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.get_optimizer_params(base_lr=1e-4, proj_lr=1e-3, cls_lr=1e-3))
        model, history_stage1, _ = train_model(
            model, train_loader, test_loader,
            optimizer, nn.CrossEntropyLoss(),
            model_name=f'{model_tag}_Stage1', num_epochs=10  # 场景 2：10 轮
        )

        # 阶段 2：解冻 20% 层
        unfreeze_ratio = 0.2
        for m in model.base_models:
            num_layers = len(list(m.parameters()))
            num_unfreeze = int(num_layers * unfreeze_ratio)
            for i, param in enumerate(m.parameters()):
                if i >= num_layers - num_unfreeze:
                    param.requires_grad = True

        optimizer = optim.AdamW(
            model.get_optimizer_params(base_lr=1e-4, proj_lr=1e-3, cls_lr=1e-3),
            weight_decay=1e-5
        )
        model, history_stage2, metrics = train_model(
            model, train_loader, test_loader,
            optimizer, nn.CrossEntropyLoss(),
            model_name=f'{model_tag}_Stage2', num_epochs=10  # 场景 2：10 轮
        )

        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_tag}_weights.pth'))
        plot_roc_curve(model, test_loader, output_dir, model_name=model_tag)

        full_history = {}
        expected_keys = ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'val_f1', 'train_time', 'inference_time']
        for key in expected_keys:
            full_history[key] = (
                history_stage1.get(key, []) + history_stage2.get(key, [])
            )

        save_results(model_tag, param_count, metrics, full_history, output_dir)
        results[model_tag] = (metrics, full_history)
    return results

def main():
    """主训练流程"""
    output_dir = "results"
    train_loader, test_loader = load_data(dataset_name='cifar10')

    # 训练基线模型
    baseline_results = train_baselines(train_loader, test_loader, output_dir=os.path.join(output_dir, "baselines"))

    # 训练融合模型
    fusion_results = train_fusions(train_loader, test_loader, output_dir=os.path.join(output_dir, "fusions"))

    # 综合可视化
    all_results = {**baseline_results, **fusion_results}
    all_histories = [h for _, h in all_results.values()]
    all_names = list(all_results.keys())
    plot_training_history(all_histories, all_names, baseline_results, fusion_results, output_dir)

if __name__ == "__main__":
    main()
