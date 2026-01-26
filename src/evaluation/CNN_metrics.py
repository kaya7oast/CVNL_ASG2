"""
Evaluation Metrics and Analysis Functions
"""

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support
)


def evaluate_model(test_preds, test_labels, test_probs, class_names, confidence_threshold=0.85):
    """
    Comprehensive model evaluation
    
    Args:
        test_preds: Predicted class indices
        test_labels: True class indices
        test_probs: Prediction probabilities
        class_names: List of class names
        confidence_threshold: Confidence threshold for operational requirements
        
    Returns:
        dict: Evaluation results
    """
    print("="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)

    # Generate classification report
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        digits=4
    )
    print("\n" + report)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, average=None
    )

    # Calculate accuracy
    accuracy = 100.0 * np.sum(test_preds == test_labels) / len(test_labels)

    # Confidence statistics
    max_probs = test_probs.max(axis=1)
    high_confidence = (max_probs >= confidence_threshold).sum()
    low_confidence = (max_probs < confidence_threshold).sum()

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'high_confidence_count': high_confidence,
        'low_confidence_count': low_confidence,
        'avg_confidence': max_probs.mean(),
        'median_confidence': np.median(max_probs),
        'report': report
    }

    return results


def calculate_confidence_stats(test_probs, confidence_threshold=0.85):
    """
    Calculate confidence statistics
    
    Args:
        test_probs: Prediction probabilities
        confidence_threshold: Confidence threshold
        
    Returns:
        dict: Confidence statistics
    """
    max_probs = test_probs.max(axis=1)
    high_confidence = (max_probs >= confidence_threshold).sum()
    low_confidence = (max_probs < confidence_threshold).sum()
    
    print(f"\nConfidence Analysis (Threshold: {confidence_threshold}):")
    print(f"  High confidence predictions (≥{confidence_threshold}): {high_confidence} ({100*high_confidence/len(test_probs):.1f}%)")
    print(f"  Low confidence predictions (<{confidence_threshold}): {low_confidence} ({100*low_confidence/len(test_probs):.1f}%)")
    print(f"  Average confidence: {max_probs.mean():.3f}")
    print(f"  Median confidence: {np.median(max_probs):.3f}")
    
    return {
        'max_probs': max_probs,
        'high_confidence': high_confidence,
        'low_confidence': low_confidence,
        'avg_confidence': max_probs.mean(),
        'median_confidence': np.median(max_probs)
    }


def analyze_misclassifications(test_preds, test_labels, class_names, confusion_mat):
    """
    Identify and analyze critical misclassifications
    
    Args:
        test_preds: Predicted class indices
        test_labels: True class indices
        class_names: List of class names
        confusion_mat: Confusion matrix
        
    Returns:
        list: Critical misclassifications
    """
    print("\n" + "="*80)
    print("CRITICAL MISCLASSIFICATION ANALYSIS")
    print("="*80)
    
    num_classes = len(class_names)
    cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    
    critical_misclassifications = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_mat[i, j] > 0:
                true_class = class_names[i]
                pred_class = class_names[j]
                count = confusion_mat[i, j]
                percentage = cm_normalized[i, j] * 100

                # Critical: Boeing 777 ↔ Airbus A330 (different fuel requirements)
                if (true_class == 'B777' and pred_class == 'A330') or \
                   (true_class == 'A330' and pred_class == 'B777'):
                    impact = "CRITICAL - Wrong fuel quantity (B777: 171,000L vs A330: 139,000L)"
                    critical_misclassifications.append({
                        'true': true_class, 'pred': pred_class,
                        'count': count, 'pct': percentage, 'impact': impact
                    })

                # High Risk: A380 misclassification (special handling)
                elif true_class == 'A380' or pred_class == 'A380':
                    impact = "HIGH RISK - A380 requires dual boarding bridges and special equipment"
                    critical_misclassifications.append({
                        'true': true_class, 'pred': pred_class,
                        'count': count, 'pct': percentage, 'impact': impact
                    })

    # Print critical misclassifications
    if critical_misclassifications:
        for idx, mc in enumerate(critical_misclassifications, 1):
            print(f"\n{idx}. {mc['true']} → {mc['pred']}")
            print(f"   Count: {mc['count']} ({mc['pct']:.1f}% of {mc['true']} samples)")
            print(f"   Impact: {mc['impact']}")
            print(f"   Mitigation: Cross-check with flight plan data")
    else:
        print("\n✓ No critical misclassifications detected")

    print("\n" + "="*80)
    
    return critical_misclassifications


def print_operational_impact(precision, recall, f1, class_names, operational_params):
    """
    Print operational impact analysis
    
    Args:
        precision: Precision scores per class
        recall: Recall scores per class
        f1: F1 scores per class
        class_names: List of class names
        operational_params: Operational parameters dict
    """
    print("\n" + "="*80)
    print("OPERATIONAL IMPACT ANALYSIS")
    print("="*80)
    
    print(f"\n{'Aircraft':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Traffic%':<10} {'Risk Level':<12}")
    print("-" * 80)

    critical_classes = []
    for i, class_name in enumerate(class_names):
        params = operational_params.get(class_name, {'traffic': 0, 'risk': 'Unknown'})
        print(f"{class_name:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} "
              f"{params['traffic']:<10}% {params['risk']:<12}")

        # Identify critical misclassifications
        if params['risk'] == 'Critical' and (precision[i] < 0.90 or recall[i] < 0.95):
            critical_classes.append(class_name)

    if critical_classes:
        print(f"\nCRITICAL ATTENTION REQUIRED:")
        for cls in critical_classes:
            print(f"   - {cls}: Precision/Recall below safety thresholds")
            print(f"     Required: Precision >90%, Recall >95% (special handling aircraft)")
    else:
        print(f"\n✓ All critical aircraft classes meet safety thresholds")
