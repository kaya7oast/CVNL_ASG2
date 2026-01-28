"""
Focal Loss Implementation for Class Imbalance
Addresses MD-11/B737 confusion by reweighting rare classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Formula: FL = -α(1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Class weights (higher for rare classes like MD-11)
        gamma: Focusing parameter (default: 2)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights_for_focal_loss(train_dataset, num_classes=8):
    """
    Calculate inverse frequency weights for focal loss
    
    Returns class weights where rare classes get higher weights
    Example: MD-11 (33 samples) gets higher weight than B737 (233 samples)
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    
    for _, label in train_dataset:
        class_counts[label] += 1
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\n" + "="*60)
    print("FOCAL LOSS CLASS WEIGHTS")
    print("="*60)
    class_names = ['A320', 'A330', 'A380', 'ATR72', 'B737', 'B777', 'CRJ900', 'MD11']
    for idx, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        print(f"{name:8s}: {int(count):4d} samples → weight: {weight:.4f}")
    print("="*60)
    
    return class_weights.tolist()


# Example usage in training loop
"""
from src.training.CNN_focal_loss import FocalLoss, get_class_weights_for_focal_loss

# Calculate class weights from training data
class_weights = get_class_weights_for_focal_loss(train_dataset)

# Create focal loss with alpha weights and gamma=2
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# Use in training
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
"""
