"""
author: Bruce Zhao
date: 2025/5/21
可视化模块，支持训练历史、混淆矩阵和 ROC 曲线，支持中文显示
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

# 配置中文字体
plt.rcParams['font.family'] = ['SimHei', 'Arial', 'sans-serif']  # 优先使用 SimHei，兼容 Arial
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def save_results(model_name, param_count, metrics, history, output_dir):
    """保存训练结果为 JSON，处理 NumPy 数组"""
    # 深拷贝 metrics 以避免修改原始数据
    metrics_copy = metrics.copy()
    # 将 confusion_matrix 转换为 Python 列表
    if 'confusion_matrix' in metrics_copy and isinstance(metrics_copy['confusion_matrix'], np.ndarray):
        metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()

    results = {
        'name': model_name,
        'parameters_million': param_count,
        'final_metrics': metrics_copy,
        'history': history
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def plot_training_history(histories, names, baseline_results, fusion_results, output_dir):
    """绘制训练历史和混淆矩阵"""
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss']
    metric_names = {
        'train_acc': '训练准确率',
        'val_acc': '验证准确率',
        'train_loss': '训练损失',
        'val_loss': '验证损失'
    }

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for history, name in zip(histories, names):
            plt.plot(history[metric], label=f'{name} {metric_names[metric]}')
        plt.title(f'{metric_names[metric]}变化曲线')
        plt.xlabel('轮次')
        plt.ylabel(metric_names[metric])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()

    for name, (metrics, _) in {**baseline_results, **fusion_results}.items():
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{name} 混淆矩阵')
        plt.colorbar()
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
        plt.close()

def plot_roc_curve(model, test_loader, output_dir, model_name="model"):
    """绘制 ROC 曲线"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for i in range(all_probs.shape[1]):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'类别 {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} ROC 曲线')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
    plt.close()
