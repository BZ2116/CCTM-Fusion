"""
author: Bruce Zhao
date: 2025/5/18
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import time
import numpy as np


def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=5, model_name="model"):
    """
    训练模型，记录多种性能指标。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'val_f1': [], 'train_time': [], 'inference_time': []
    }
    confusion_matrices = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [{model_name}] Training')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_time = time.time() - start_time
        avg_train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['train_time'].append(epoch_time)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        inference_start = time.time()
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch + 1} [{model_name}] Evaluating')
        with torch.no_grad():
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        inference_time = (time.time() - inference_start) / len(test_loader)
        avg_val_loss = val_loss / len(test_loader)
        val_acc = accuracy_score(all_labels, all_preds)  # Fixed
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['inference_time'].append(inference_time)
        confusion_matrices.append(cm)

        print(f'Epoch {epoch + 1}, {model_name} | '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}, Train Time: {epoch_time:.2f}s, '
              f'Inference Time: {inference_time:.4f}s')

    final_metrics = {
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_loss': avg_val_loss,
        'avg_train_time': np.mean(history['train_time']),
        'avg_inference_time': np.mean(history['inference_time']),
        'confusion_matrix': cm
    }
    return model, history, final_metrics
