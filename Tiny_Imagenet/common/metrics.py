import torch
import numpy as np


def calculate_class_distances(features, labels):
    """
    Calculate average intra-class and inter-class distances.
    
    Args:
        features: Feature vectors (tensor)
        labels: Class labels (tensor)
        
    Returns:
        tuple: (average intra-class distance, average inter-class distance)
    """
    unique_classes = torch.unique(labels)
    class_means = []
    intra_class_distances = []
    
    # Calculate class centroids and intra-class distances
    for cls in unique_classes:
        cls_features = features[labels == cls]
        cls_mean = cls_features.mean(dim=0)
        class_means.append(cls_mean)
        
        # Average distance from each point to its class centroid
        dists = torch.norm(cls_features - cls_mean, dim=1).mean()
        intra_class_distances.append(dists.item())
    
    # Convert to tensor for easier computation
    class_means = torch.stack(class_means)
    
    # Calculate inter-class distances (between centroids)
    n_classes = len(unique_classes)
    inter_dists = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            dist = torch.norm(class_means[i] - class_means[j])
            inter_dists.append(dist.item())
    
    avg_intra_dist = np.mean(intra_class_distances)
    avg_inter_dist = np.mean(inter_dists) if inter_dists else 0
    return avg_intra_dist, avg_inter_dist


def calculate_feature_metrics(features_before, features_after, labels):
    """
    Calculate metrics to assess feature quality.
    
    Args:
        features_before: Features before RAMA application
        features_after: Features after RAMA application
        labels: Class labels
        
    Returns:
        dict: Feature quality metrics
    """
    metrics = {}
    
    # 1. Calculate intra-class and inter-class distances before RAMA
    intra_dist_before, inter_dist_before = calculate_class_distances(features_before, labels)
    
    # 2. Calculate intra-class and inter-class distances after RAMA
    intra_dist_after, inter_dist_after = calculate_class_distances(features_after, labels)
    
    # 3. Fisher's criterion (inter-class separation / intra-class spread)
    fisher_before = inter_dist_before / (intra_dist_before + 1e-8)
    fisher_after = inter_dist_after / (intra_dist_after + 1e-8)
    
    metrics['intra_class_distance_before'] = intra_dist_before
    metrics['inter_class_distance_before'] = inter_dist_before
    metrics['intra_class_distance_after'] = intra_dist_after
    metrics['inter_class_distance_after'] = inter_dist_after
    metrics['fisher_ratio_before'] = fisher_before
    metrics['fisher_ratio_after'] = fisher_after
    metrics['fisher_improvement'] = fisher_after / fisher_before
    
    return metrics