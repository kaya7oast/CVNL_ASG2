"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualize which regions the model focuses on for MD-11 vs B737 classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN attention
    
    Shows which image regions contribute most to classification decision
    Useful for debugging MD-11/B737 confusion
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained CNN model
            target_layer: Layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, image_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for specific class
        
        Args:
            image_tensor: Input image [1, 3, H, W]
            class_idx: Target class index (None = predicted class)
            
        Returns:
            cam: Heatmap showing important regions [H, W]
            prediction: Model prediction
        """
        # Forward pass
        self.model.eval()
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU (only positive influence)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def visualize_comparison(self, image_path, true_class, predicted_class, class_names, save_path=None):
        """
        Visualize what model looks at for true class vs predicted class
        
        Useful for understanding MD-11 → B737 misclassification
        """
        # Load and preprocess image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            self.model = self.model.cuda()
        
        # Generate CAMs for both classes
        cam_true, _ = self.generate_cam(image_tensor, class_idx=true_class)
        cam_pred, _ = self.generate_cam(image_tensor, class_idx=predicted_class)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (224, 224))
        axes[0].imshow(orig_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # CAM for true class
        heatmap_true = cv2.applyColorMap(np.uint8(255 * cam_true), cv2.COLORMAP_JET)
        heatmap_true = cv2.cvtColor(heatmap_true, cv2.COLOR_BGR2RGB)
        heatmap_true = cv2.resize(heatmap_true, (224, 224))
        overlay_true = cv2.addWeighted(orig_img, 0.5, heatmap_true, 0.5, 0)
        axes[1].imshow(overlay_true)
        axes[1].set_title(f'True Class: {class_names[true_class]}')
        axes[1].axis('off')
        
        # CAM for predicted class
        heatmap_pred = cv2.applyColorMap(np.uint8(255 * cam_pred), cv2.COLORMAP_JET)
        heatmap_pred = cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB)
        heatmap_pred = cv2.resize(heatmap_pred, (224, 224))
        overlay_pred = cv2.addWeighted(orig_img, 0.5, heatmap_pred, 0.5, 0)
        axes[2].imshow(overlay_pred)
        axes[2].set_title(f'Predicted: {class_names[predicted_class]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig


def analyze_md11_confusion(model, md11_image_path, output_dir='outputs/gradcam'):
    """
    Analyze why model confuses MD-11 with B737
    
    Args:
        model: Trained model
        md11_image_path: Path to misclassified MD-11 image
        output_dir: Where to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get last convolutional layer (for ResNet50: layer4)
    if hasattr(model, 'backbone'):
        target_layer = model.backbone.layer4[-1]
    else:
        # If model is just the ResNet
        target_layer = model.layer4[-1]
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Class names
    class_names = ['A320', 'A330', 'A380', 'ATR72', 'B737', 'B777', 'CRJ900', 'MD11']
    
    # MD11 = index 7, B737 = index 4
    md11_idx = 7
    b737_idx = 4
    
    # Generate visualization
    save_path = os.path.join(output_dir, 'md11_vs_b737_confusion.png')
    gradcam.visualize_comparison(
        md11_image_path, 
        true_class=md11_idx,
        predicted_class=b737_idx,
        class_names=class_names,
        save_path=save_path
    )
    
    print("\n" + "="*60)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("="*60)
    print(f"Visualization saved to: {save_path}")
    print("\nInterpretation Guide:")
    print("- RED regions: Strong activation (model focuses here)")
    print("- BLUE regions: Weak activation (model ignores)")
    print("\nCompare MD11 vs B737 heatmaps:")
    print("- If both focus on background → increase CutOut augmentation")
    print("- If missing engines/tail → add region-specific attention")
    print("- If confused by angle → add more viewpoint augmentation")
    print("="*60)


# Example usage
"""
from src.evaluation.CNN_gradcam import analyze_md11_confusion

# Load trained model
model = torch.load('outputs/models/best_model_phase2.pth')

# Analyze a misclassified MD-11 image
analyze_md11_confusion(
    model, 
    md11_image_path='path/to/misclassified_md11.jpg',
    output_dir='outputs/gradcam'
)
"""
