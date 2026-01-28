"""
MD-11 Specific Data Augmentation
Addresses domain generalization issue for MD-11 class
"""

import torch
from torchvision import transforms
import numpy as np
from PIL import ImageFilter
import random


class MD11AggressiveAugmentation:
    """
    Aggressive augmentation strategy specifically for MD-11 class
    
    Goals:
    1. 10x synthetic sample expansion (33 → ~300 effective samples)
    2. Force model to learn MD-11 invariant features
    3. Reduce reliance on FGVC-specific background cues
    """
    
    def __init__(self):
        self.augmentation = transforms.Compose([
            # Extreme rotation for varying camera angles
            transforms.RandomRotation(degrees=25),
            
            # Multi-scale training
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            
            # Advanced color transforms
            transforms.ColorJitter(
                brightness=0.4,  # ±40% vs standard ±30%
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Random blur for focus variations
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            
            # Random grayscale for haze conditions
            transforms.RandomGrayscale(p=0.15),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Background randomization (CutOut simulation)
            RandomCutout(n_holes=3, length=20, p=0.4),
            
            # Normalize
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, image):
        return self.augmentation(image)


class RandomCutout:
    """
    Randomly mask out square regions to prevent background overfitting
    Forces model to learn from partial views
    """
    
    def __init__(self, n_holes=1, length=16, p=0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img: Tensor image [C, H, W]
        """
        if random.random() > self.p:
            return img
        
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask).unsqueeze(0)
        img = img * mask
        
        return img


def create_md11_balanced_loader(train_dataset, batch_size=32, md11_oversample_factor=7):
    """
    Create data loader with MD-11 oversampling to balance class distribution
    
    Args:
        train_dataset: Original training dataset
        batch_size: Batch size
        md11_oversample_factor: How many times to oversample MD-11 (to match B737)
        
    Returns:
        Balanced data loader with MD-11 augmented samples
    """
    from torch.utils.data import WeightedRandomSampler, DataLoader
    
    # Calculate sample weights
    class_counts = torch.zeros(8)
    for _, label in train_dataset:
        class_counts[label] += 1
    
    # MD-11 is class index 7
    md11_idx = 7
    
    # Create per-sample weights
    sample_weights = []
    for _, label in train_dataset:
        if label == md11_idx:
            # Oversample MD-11
            weight = md11_oversample_factor
        else:
            weight = 1.0
        sample_weights.append(weight)
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loader
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print("MD-11 BALANCED DATA LOADER CREATED")
    print("="*60)
    print(f"Original MD-11 samples: {int(class_counts[md11_idx])}")
    print(f"Oversample factor: {md11_oversample_factor}x")
    print(f"Effective MD-11 samples per epoch: ~{int(class_counts[md11_idx] * md11_oversample_factor)}")
    print(f"B737 samples: {int(class_counts[4])} (for comparison)")
    print("="*60)
    
    return loader


def apply_md11_specific_augmentation(train_dataset):
    """
    Apply MD-11 specific augmentation to training dataset
    
    Modifies the dataset transform for MD-11 samples only
    """
    original_transform = train_dataset.transform
    md11_transform = MD11AggressiveAugmentation()
    
    # Class index for MD-11
    md11_idx = 7
    
    class MD11AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, md11_transform, original_transform):
            self.base_dataset = base_dataset
            self.md11_transform = md11_transform
            self.original_transform = original_transform
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            # Get original item
            img_path = self.base_dataset.samples[idx][0]
            label = self.base_dataset.samples[idx][1]
            img = self.base_dataset.loader(img_path)
            
            # Apply different transforms based on class
            if label == md11_idx:
                # MD-11: aggressive augmentation
                img = self.md11_transform(img)
            else:
                # Other classes: standard augmentation
                img = self.original_transform(img)
            
            return img, label
    
    augmented_dataset = MD11AugmentedDataset(
        train_dataset, 
        md11_transform, 
        original_transform
    )
    
    print("\n" + "="*60)
    print("MD-11 SPECIFIC AUGMENTATION APPLIED")
    print("="*60)
    print("MD-11 samples receive aggressive augmentation:")
    print("  • Rotation: ±25° (vs ±15° standard)")
    print("  • Scale: 0.6-1.0 (vs 0.7-1.0 standard)")
    print("  • Brightness: ±40% (vs ±30% standard)")
    print("  • CutOut: 3 holes (background randomization)")
    print("  • Blur: 30% probability")
    print("Other classes use standard augmentation")
    print("="*60)
    
    return augmented_dataset


# Example usage in training pipeline
"""
from CNN_md11_augmentation import apply_md11_specific_augmentation, create_md11_balanced_loader

# Load original dataset
train_dataset = torchvision.datasets.ImageFolder(
    'data/train',
    transform=standard_transform
)

# Apply MD-11 specific augmentation
augmented_dataset = apply_md11_specific_augmentation(train_dataset)

# Create balanced loader with MD-11 oversampling
train_loader = create_md11_balanced_loader(
    augmented_dataset,
    batch_size=32,
    md11_oversample_factor=7  # Match B737 frequency
)

# Train with focal loss
from src.training.CNN_focal_loss import FocalLoss, get_class_weights_for_focal_loss

class_weights = get_class_weights_for_focal_loss(train_dataset)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
"""
