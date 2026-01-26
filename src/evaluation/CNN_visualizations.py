"""
Visualization Functions for Training and Evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torchvision.transforms import functional as TF


def plot_confusion_matrix(test_labels, test_preds, class_names, save_path='outputs/plots/confusion_matrix.png'):
    """
    Plot confusion matrix (raw counts and normalized)
    
    Args:
        test_labels: True labels
        test_preds: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualize confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)

    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confusion matrix saved to {save_path}")
    
    return cm, cm_normalized


def plot_training_history(history_phase1, history_phase2, best_val_acc_phase2, 
                          save_path='outputs/plots/training_history.png'):
    """
    Plot training history for both phases
    
    Args:
        history_phase1: Phase 1 training history
        history_phase2: Phase 2 training history
        best_val_acc_phase2: Best validation accuracy from phase 2
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Phase 1 Loss
    axes[0, 0].plot(history_phase1['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(history_phase1['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_title('Phase 1: Loss (Frozen Backbone)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase 1 Accuracy
    axes[0, 1].plot(history_phase1['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[0, 1].plot(history_phase1['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[0, 1].axhline(y=85, color='r', linestyle='--', label='ICAO Standard (85%)', alpha=0.7)
    axes[0, 1].set_title('Phase 1: Accuracy (Frozen Backbone)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Phase 2 Loss
    axes[1, 0].plot(history_phase2['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1, 0].plot(history_phase2['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[1, 0].set_title('Phase 2: Loss (Fine-Tuning)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Phase 2 Accuracy
    axes[1, 1].plot(history_phase2['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[1, 1].plot(history_phase2['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[1, 1].axhline(y=85, color='r', linestyle='--', label='ICAO Standard (85%)', alpha=0.7)
    axes[1, 1].axhline(y=best_val_acc_phase2, color='g', linestyle=':',
                       label=f'Best: {best_val_acc_phase2:.2f}%', alpha=0.7)
    axes[1, 1].set_title('Phase 2: Accuracy (Fine-Tuning)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Two-Phase Training Strategy: Changi AeroVision',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Combined accuracy plot
    plt.figure(figsize=(14, 6))
    total_epochs_p1 = len(history_phase1['val_acc'])
    total_epochs_p2 = len(history_phase2['val_acc'])

    # Plot Phase 1
    plt.plot(range(1, total_epochs_p1 + 1),
             history_phase1['val_acc'],
             marker='o', linewidth=2.5, label='Phase 1 (Frozen Backbone)', color='blue')

    # Plot Phase 2
    plt.plot(range(total_epochs_p1 + 1, total_epochs_p1 + total_epochs_p2 + 1),
             history_phase2['val_acc'],
             marker='s', linewidth=2.5, label='Phase 2 (Fine-Tuning)', color='green')

    # Add phase separator
    plt.axvline(x=total_epochs_p1, color='red', linestyle='--', linewidth=2,
                label='Phase Transition', alpha=0.7)

    # Add benchmarks
    plt.axhline(y=85, color='orange', linestyle='--', label='ICAO Standard (85%)', alpha=0.7)
    plt.axhline(y=best_val_acc_phase2, color='purple', linestyle=':',
                label=f'Best Achieved: {best_val_acc_phase2:.2f}%', alpha=0.7, linewidth=2)

    plt.title('Validation Accuracy: Two-Phase Training Strategy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_path = save_path.replace('training_history', 'combined_accuracy')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training history plots saved")


def plot_business_impact(annual_movements, avg_delay_reduction, cost_per_delay_minute,
                        save_path='outputs/plots/business_impact.png'):
    """
    Plot business impact analysis
    
    Args:
        annual_movements: Annual aircraft movements
        avg_delay_reduction: Average delay reduction in minutes
        cost_per_delay_minute: Cost per delay minute in SGD
        save_path: Path to save the plot
    """
    # Calculate operational savings
    annual_time_saved_hours = (annual_movements * avg_delay_reduction) / 60
    annual_cost_saving = annual_movements * avg_delay_reduction * cost_per_delay_minute
    additional_flights_daily = annual_time_saved_hours / 24 / 365

    # Visualize business impact
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time savings
    categories = ['Current\nManual\nProcess', 'With\nAeroVision']
    time_data = [annual_movements * 2.5 / 60, 0]  # Hours wasted
    axes[0].bar(categories, [time_data[0], 0], color=['#e74c3c', '#2ecc71'], alpha=0.7)
    axes[0].set_ylabel('Annual Delay Hours', fontsize=11)
    axes[0].set_title(f'Time Saved: {annual_time_saved_hours:,.0f} Hours/Year',
                     fontsize=12, fontweight='bold')
    axes[0].text(0, time_data[0]/2, f'{time_data[0]:,.0f}h\nwasted',
                ha='center', fontsize=10, fontweight='bold')
    axes[0].text(1, time_data[0]/2, f'{annual_time_saved_hours:,.0f}h\nSAVED',
                ha='center', fontsize=10, fontweight='bold', color='green')

    # Cost savings
    cost_data = [annual_cost_saving, 0]
    axes[1].bar(categories, [cost_data[0], 0], color=['#e74c3c', '#2ecc71'], alpha=0.7)
    axes[1].set_ylabel('Annual Cost (SGD Million)', fontsize=11)
    axes[1].set_title(f'Cost Saved: SGD {annual_cost_saving/1e6:.1f}M/Year',
                     fontsize=12, fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    axes[1].text(0, cost_data[0]/2, f'SGD {annual_cost_saving/1e6:.0f}M\nlost',
                ha='center', fontsize=10, fontweight='bold')
    axes[1].text(1, cost_data[0]/2, f'SGD {annual_cost_saving/1e6:.0f}M\nSAVED',
                ha='center', fontsize=10, fontweight='bold', color='green')

    # Capacity increase
    current_capacity = 100  # Baseline
    new_capacity = current_capacity + (additional_flights_daily / 1000 * 100)  # Percentage increase
    axes[2].bar(categories, [current_capacity, new_capacity],
               color=['#3498db', '#2ecc71'], alpha=0.7)
    axes[2].set_ylabel('Relative Capacity (%)', fontsize=11)
    axes[2].set_title(f'Capacity Increase: +{additional_flights_daily:.1f} Flights/Day',
                     fontsize=12, fontweight='bold')
    axes[2].text(0, current_capacity/2, f'{current_capacity}%\nbaseline',
                ha='center', fontsize=10, fontweight='bold')
    axes[2].text(1, new_capacity/2, f'+{additional_flights_daily:.1f}\nflights/day',
                ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Changi AeroVision: Business Impact Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Business impact plot saved to {save_path}")


def plot_inference_time(inference_times, operational_requirement=100,
                       save_path='outputs/plots/inference_time.png'):
    """
    Plot inference time distribution and stability
    
    Args:
        inference_times: List of inference times in milliseconds
        operational_requirement: Operational requirement in ms
        save_path: Path to save the plot
    """
    avg_inference_time = np.mean(inference_times)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(inference_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(avg_inference_time, color='red', linestyle='--',
               label=f'Mean: {avg_inference_time:.2f}ms', linewidth=2)
    plt.axvline(operational_requirement, color='green', linestyle='--',
               label=f'Requirement: <{operational_requirement}ms', linewidth=2)
    plt.xlabel('Inference Time (ms)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Inference Time Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(inference_times, linewidth=1, alpha=0.7)
    plt.axhline(avg_inference_time, color='red', linestyle='--',
               label=f'Mean: {avg_inference_time:.2f}ms', linewidth=2)
    plt.axhline(operational_requirement, color='green', linestyle='--',
               label=f'Requirement: <{operational_requirement}ms', linewidth=2)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Inference Time (ms)', fontsize=11)
    plt.title('Inference Time Stability', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Inference time plot saved to {save_path}")


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def plot_sample_predictions(model, test_loader, idx_to_class, device, num_samples=12,
                            save_path='outputs/plots/sample_predictions.png'):
    """
    Visualize sample predictions
    
    Args:
        model: Trained model
        test_loader: Test data loader
        idx_to_class: Index to class name mapping
        device: Device
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    import torch.nn.functional as F
    
    model.eval()
    
    # Get a batch from test set
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)

    # Visualize predictions
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()

    for idx in range(num_samples):
        img = denormalize_image(images[idx].cpu())
        img = img.permute(1, 2, 0).numpy()

        true_label = idx_to_class[labels[idx].item()]
        pred_label = idx_to_class[predictions[idx].item()]
        confidence = confidences[idx].item()

        # Determine if prediction is correct
        is_correct = (labels[idx] == predictions[idx]).item()
        color = 'green' if is_correct else 'red'

        axes[idx].imshow(img)
        axes[idx].axis('off')

        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"
        axes[idx].set_title(title, fontsize=10, fontweight='bold', color=color)

        # Add border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.suptitle('Sample Predictions: Changi AeroVision\n(Green = Correct, Red = Incorrect)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Sample predictions saved to {save_path}")
