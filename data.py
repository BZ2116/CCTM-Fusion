"""
author:Bruce Zhao
date: 2025/5/18
"""
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def get_data_loaders(batch_size=32, data_dir='./data', input_size=224, normalize=None):
    """
    加载CIFAR-10数据集，应用预处理和数据增强，返回训练和测试数据加载器。

    参数:
        batch_size (int): 批量大小
        data_dir (str): 数据存储目录
        input_size (int): 输入图像尺寸
        normalize (dict): 归一化参数，包含mean和std

    返回:
        train_loader, test_loader: 训练和测试数据加载器
    """
    if normalize is None:
        normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std']),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std']),
    ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
