"""
author:Bruce Zhao
date: 2025/5/18
数据加载模块，支持 CIFAR-10 和医疗图像数据集
"""


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(dataset_name='cifar10', batch_size=8, input_size=224, normalize=None):
    """加载指定数据集，支持 CIFAR-10 和其他数据集"""
    if normalize is None:
        normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # 医疗图像增强
    medical_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
    ])

    # 非医疗图像增强
    standard_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
    ])

    # 测试集变换
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
    ])

    # 选择数据集
    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=standard_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=standard_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name.lower() == 'svhn':
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=standard_transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please implement loading logic for {dataset_name}.")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
