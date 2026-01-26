"""
Training and Validation Functions
Two-phase training strategy with frozen backbone and fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def train_epoch(model, dataloader, criterion, optimizer, device, phase_name="Training"):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        phase_name: Name of training phase for progress bar
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"{phase_name}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, phase_name="Validation"):
    """
    Validate the model
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        phase_name: Name of validation phase for progress bar
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy, predictions, labels, probabilities)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{phase_name}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get probabilities
            probs = F.softmax(outputs, dim=1)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_two_phase(model, train_loader, val_loader, config, device):
    """
    Two-phase training strategy
    Phase 1: Train classifier head only (frozen backbone)
    Phase 2: Fine-tune entire network
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object with hyperparameters
        device: Device to train on
        
    Returns:
        dict: Training history and best model information
    """
    criterion = nn.CrossEntropyLoss()
    
    # ========== PHASE 1: Train Classifier Head Only ==========
    print("="*80)
    print("PHASE 1: TRAINING CLASSIFIER HEAD (FROZEN BACKBONE)")
    print("="*80)
    print(f"Epochs: {config.EPOCHS_PHASE1}")
    print(f"Learning rate: {config.LEARNING_RATE_PHASE1}")
    print(f"Strategy: Train only final classifier layers while backbone remains frozen")
    print("="*80)

    # Freeze backbone
    model.freeze_backbone()

    # Phase 1 optimizer
    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE_PHASE1,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1, mode='min', factor=0.5, patience=2
    )

    # Training history
    history_phase1 = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc_phase1 = 0.0

    # Training loop for Phase 1
    for epoch in range(1, config.EPOCHS_PHASE1 + 1):
        print(f"\n{'='*80}")
        print(f"PHASE 1 - Epoch {epoch}/{config.EPOCHS_PHASE1}")
        print(f"{'='*80}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_phase1, device,
            phase_name=f"Phase 1 Train Epoch {epoch}"
        )

        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, criterion, device,
            phase_name=f"Phase 1 Val Epoch {epoch}"
        )

        # Update scheduler
        scheduler_phase1.step(val_loss)

        # Save history
        history_phase1['train_loss'].append(train_loss)
        history_phase1['train_acc'].append(train_acc)
        history_phase1['val_loss'].append(val_loss)
        history_phase1['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc_phase1:
            best_val_acc_phase1 = val_acc
            torch.save(model.state_dict(), f'{config.MODELS_DIR}/best_model_phase1.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc_phase1:.2f}%")
    print("="*80)

    # ========== PHASE 2: Fine-Tune Entire Network ==========
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING ENTIRE NETWORK")
    print("="*80)
    print(f"Epochs: {config.EPOCHS_PHASE2}")
    print(f"Learning rate: {config.LEARNING_RATE_PHASE2} (10x lower than Phase 1)")
    print(f"Strategy: Gradual unfreezing with lower learning rate")
    print("="*80)

    # Unfreeze backbone for fine-tuning
    model.unfreeze_backbone(num_layers=15)

    # Phase 2 optimizer with lower learning rate
    optimizer_phase2 = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE_PHASE2,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=config.EPOCHS_PHASE2, eta_min=1e-6
    )

    # Training history
    history_phase2 = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc_phase2 = best_val_acc_phase1
    best_epoch = 0

    # Training loop for Phase 2
    for epoch in range(1, config.EPOCHS_PHASE2 + 1):
        print(f"\n{'='*80}")
        print(f"PHASE 2 - Epoch {epoch}/{config.EPOCHS_PHASE2}")
        print(f"Current LR: {optimizer_phase2.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_phase2, device,
            phase_name=f"Phase 2 Train Epoch {epoch}"
        )

        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, criterion, device,
            phase_name=f"Phase 2 Val Epoch {epoch}"
        )

        # Update scheduler
        scheduler_phase2.step()

        # Save history
        history_phase2['train_loss'].append(train_loss)
        history_phase2['train_acc'].append(train_acc)
        history_phase2['val_loss'].append(val_loss)
        history_phase2['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'{config.MODELS_DIR}/best_model_phase2.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

        # Early stopping check (optional)
        if epoch - best_epoch > 5:
            print(f"\n  Early stopping triggered (no improvement for 5 epochs)")
            break

    print("\n" + "="*80)
    print("PHASE 2 COMPLETE - TRAINING FINISHED")
    print("="*80)
    print(f"Phase 1 Best Validation Accuracy: {best_val_acc_phase1:.2f}%")
    print(f"Phase 2 Best Validation Accuracy: {best_val_acc_phase2:.2f}%")
    print(f"Improvement: {best_val_acc_phase2 - best_val_acc_phase1:.2f}%")
    print("="*80)

    return {
        'history_phase1': history_phase1,
        'history_phase2': history_phase2,
        'best_val_acc_phase1': best_val_acc_phase1,
        'best_val_acc_phase2': best_val_acc_phase2,
        'criterion': criterion
    }
