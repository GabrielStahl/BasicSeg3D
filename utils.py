import matplotlib.pyplot as plt
import numpy as np
import torch

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

def calculate_dice_coefficient(pred, target, threshold=0.5):
    """
    Calculate the Dice coefficient between predicted and target segmentation masks.
    CAVE: currently returns 0.0 if an error occurs during calculation.
    """
    try:
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection) / (union + 1e-7)
        return dice.item()
    except RuntimeError as e:
        print(f"Error in calculate_dice_coefficient: {str(e)}")
        return 0.0

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return precision.item(), recall.item(), f1.item()