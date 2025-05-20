"""
author: Bruce Zhao
date: 2025/5/20
主训练流程文件 - CCTM-Fusion，限制100个数据样本
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
from data import get_data_loaders
from models import BaselineModel, FusionModel
from train import train_model
from visualize import save_results, plot_training_history
import os
import datetime

# 设置随机种子
torch.manual_seed(42)

# --------------------------
# 数据加载
# --------------------------
def load_data():
    """加载数据并限制为100个样本"""
    train_loader, test_loader = get_data_loaders(
        batch_size=16,
        input_size=224,
        normalize={
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    )

    train_indices = list(range(min(80, len(train_loader.dataset))))
    train_subset = Subset(train_loader.dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=16, shuffle=True
    )

    test_indices = list(range(min(20, len(test_loader.dataset))))
    test_subset = Subset(test_loader.dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=16, shuffle=False
    )

    print(f"Using {len(train_subset)} training samples and {len(test_subset)} test samples")
    return train_loader, test_loader

# --------------------------
# 计算参数量
# --------------------------
def count_parameters(model):
    """计算模型参数量（单位：百万）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# --------------------------
# 训练基线模型
# --------------------------
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
    weights_save_dir = os.path.join(output_dir, "saved_weights") # Directory for saved weights
    os.makedirs(weights_save_dir, exist_ok=True)

    for config in model_configs:
        name = config['name']
        weight_path = config['weight_path']
        print(f'\n=== Training Baseline: {name} ===')
        model = BaselineModel(name, weight_path, num_classes=10)

        param_count = count_parameters(model)
        print(f"Parameters: {param_count:.2f}M")

        for param in model.base_model.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW([
            {'params': model.classifier.parameters(), 'lr': 1e-3},
            {'params': model.base_model.parameters(), 'lr': 1e-5}
        ])

        # Get the trained model back
        trained_model, history, metrics = train_model(
            model, train_loader, test_loader,
            optimizer, nn.CrossEntropyLoss(),
            model_name=name
        )

        # Save model weights
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(weights_save_dir, f"{name}_{timestamp}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Saved baseline model weights to {model_save_path}")

        save_results(name, param_count, metrics, history, output_dir)
        results[name] = (metrics, history)
    return results

# --------------------------
# 训练融合模型
# --------------------------
def train_fusions(train_loader, test_loader, output_dir="results/fusions"):
    """训练融合模型"""
    model_configs = [
        {'name': 'resnet50', 'weight_path': 'weights/resnet50.bin'},
        {'name': 'vit_base_patch16_224', 'weight_path': 'weights/vit_base_patch16_224.bin'},
        {'name': 'efficientnet_b0', 'weight_path': 'weights/efficientnet_b0.bin'},
        {'name': 'mixer_b16_224', 'weight_path': 'weights/mixer_b16_224.bin'}
    ]

    combinations = [
        [model_configs[0], model_configs[1], model_configs[2], model_configs[3]],
        [model_configs[0], model_configs[1], model_configs[2]],
        [model_configs[0], model_configs[1]],
        [model_configs[0], model_configs[2]],
        [model_configs[1], model_configs[3]]
    ]

    results = {}
    os.makedirs(output_dir, exist_ok=True)
    weights_save_dir = os.path.join(output_dir, "saved_weights") # Directory for saved weights
    os.makedirs(weights_save_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        model_tag = 'CCTM-Fusion' if i == 0 else f'Fusion_{i}'
        print(f'\n=== Training Fusion: {model_tag} ===')
        model_names = [c['name'] for c in combo]
        weight_paths = [c['weight_path'] for c in combo]
        model = FusionModel(
            model_names, weight_paths, num_classes=10,
            freeze_layers_ratio=0.5, proj_dim=512, cls_hidden_dim=1024, num_heads=8
        )

        param_count = count_parameters(model)
        print(f"Parameters: {param_count:.2f}M")

        # Stage 1
        for param in model.base_models.parameters():
            param.requires_grad = False
        optimizer_s1 = optim.AdamW(model.get_optimizer_params(base_lr=1e-4, proj_lr=1e-3, cls_lr=1e-3))
        # Model is updated in-place by train_model
        model, history_stage1, _ = train_model( # Capture model
            model, train_loader, test_loader,
            optimizer_s1, nn.CrossEntropyLoss(),
            model_name=f'{model_tag}_Stage1'
        )

        # Stage 2
        unfreeze_ratio = 0.2
        for m in model.base_models:
            num_layers = len(list(m.parameters()))
            num_unfreeze = int(num_layers * unfreeze_ratio)
            for idx, param in enumerate(m.parameters()): # Changed i to idx to avoid conflict with outer loop variable
                if idx >= num_layers - num_unfreeze:
                    param.requires_grad = True

        optimizer_s2 = optim.AdamW(
            model.get_optimizer_params(base_lr=1e-4, proj_lr=1e-3, cls_lr=1e-3),
            weight_decay=1e-5
        )
        # Model is further updated in-place
        trained_model, history_stage2, metrics = train_model( # Capture final trained_model
            model, train_loader, test_loader,
            optimizer_s2, nn.CrossEntropyLoss(),
            model_name=f'{model_tag}_Stage2'
        )

        # Save model weights after Stage 2
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(weights_save_dir, f"{model_tag}_{timestamp}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Saved fusion model weights to {model_save_path}")

        full_history = {}
        expected_keys = ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'val_f1', 'train_time', 'inference_time']
        for key in expected_keys:
            full_history[key] = (
                history_stage1.get(key, []) + history_stage2.get(key, [])
            )

        save_results(model_tag, param_count, metrics, full_history, output_dir)
        results[model_tag] = (metrics, full_history)
    return results

# --------------------------
# 结果展示
# --------------------------
def display_results(baseline_results, fusion_results):
    """展示训练结果"""
    print("\n=== 最终结果 ===")
    print("--- 基线模型 ---")
    for name, (metrics, _) in baseline_results.items():
        print(f"{name:<30}: Acc={metrics['val_acc']:.4f}, F1={metrics['val_f1']:.4f}, "
              f"Loss={metrics['val_loss']:.4f}, Time={metrics['avg_train_time']:.2f}s")

    print("\n--- 融合模型 ---")
    for name, (metrics, _) in fusion_results.items():
        print(f"{name:<30}: Acc={metrics['val_acc']:.4f}, F1={metrics['val_f1']:.4f}, "
              f"Loss={metrics['val_loss']:.4f}, Time={metrics['avg_train_time']:.2f}s")

# --------------------------
# 主流程
# --------------------------
def main():
    """主训练流程"""
    output_dir = "results"
    train_loader, test_loader = load_data()
    baseline_results = train_baselines(train_loader, test_loader, output_dir=os.path.join(output_dir, "baselines"))
    fusion_results = train_fusions(train_loader, test_loader, output_dir=os.path.join(output_dir, "fusions"))
    display_results(baseline_results, fusion_results)
    all_histories = [h for _, h in baseline_results.values()] + [h for _, h in fusion_results.values()]
    all_names = list(baseline_results.keys()) + list(fusion_results.keys())
    plot_training_history(all_histories, all_names, baseline_results, fusion_results, output_dir)

if __name__ == "__main__":
    main()