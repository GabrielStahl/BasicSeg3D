import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def visualize_images(images, titles=None, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(image, cmap='gray')
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        dice_scores = []
        for class_idx in range(pred.size(1)):
            pred_class = pred[:, class_idx, ...]
            target_class = (target == class_idx).float()
            dice = (2.0 * (pred_class * target_class).sum()) / (pred_class.sum() + target_class.sum() + 1e-7)
            dice_scores.append(dice)
        return 1 - torch.mean(torch.stack(dice_scores))

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return precision.item(), recall.item(), f1.item(), dice.item()