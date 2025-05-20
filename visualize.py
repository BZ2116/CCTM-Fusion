"""
author: Bruce Zhao
date: 2025/5/18
可视化和数据保存模块，负责绘制训练历史图表和保存训练结果
"""

import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --------------------------
# 辅助函数：使数据JSON可序列化
# --------------------------
def make_json_serializable(obj):
    """递归地将NumPy数组和其他非JSON可序列化对象转换为可序列化类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj

# --------------------------
# 保存训练结果
# --------------------------
def save_results(name, param_count, metrics, history, output_dir="results"):
    """保存训练结果到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    result = {
        'name': name,
        'parameters_million': param_count,
        'final_metrics': metrics,
        'history': history
    }
    # 转换NumPy数组为列表
    result = make_json_serializable(result)
    with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
        json.dump(result, f, indent=4)
    print(f"结果已保存至 {os.path.join(output_dir, f'{name}_results.json')}")

# --------------------------
# 可视化函数
# --------------------------
def plot_training_history(histories, names, baseline_results, fusion_results, output_dir="results"):
    """绘制训练历史，生成多图，突出CCTM"""
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'val_f1', 'train_time', 'inference_time']

    # 绘制指标曲线
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for history, name in zip(histories, names):
            if metric in history and history[metric]:
                # 为 CCTM 设置更粗的线条和醒目颜色
                if name == 'CCTM':
                    plt.plot(history[metric], label=name, linewidth=3, color='red', linestyle='--')
                else:
                    plt.plot(history[metric], label=name, linewidth=1.5)
            else:
                print(f"警告: 未找到 {name} 的 {metric} 数据")
        plt.title(f'{metric.replace("_", " ").title()} 随训练轮数变化')
        plt.xlabel('训练轮数')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()

    # 绘制混淆矩阵（仅最后 epoch）
    all_results = list(baseline_results.values()) + list(fusion_results.values())
    for name, (metrics, _) in zip(names, all_results):
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'混淆矩阵 - {name}')
            plt.colorbar()
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
            plt.close()