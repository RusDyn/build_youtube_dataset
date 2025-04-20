"""
Custom loss functions for viral titles training.
"""
import torch
import torch.nn.functional as F

class PairwiseRankingLoss:
    """
    Custom implementation of a pairwise ranking loss function.
    This loss is designed to ensure that higher-scored items are ranked above lower-scored ones.
    """
    def __init__(self, margin=0.1):
        self.margin = margin
        
    def __call__(self, logits, labels):
        """
        Compute pairwise margin ranking loss.
        
        Args:
            logits: Model predictions (Tensor of shape [batch_size])
            labels: Ground truth scores (Tensor of shape [batch_size])
            
        Returns:
            loss: Pairwise ranking loss
        """
        batch_size = logits.size(0)
        if batch_size <= 1:
            # Can't do pairwise with just one sample
            return F.mse_loss(logits, labels)
            
        # Reshape for all pairwise combinations
        logits_i = logits.unsqueeze(1).expand(batch_size, batch_size)
        logits_j = logits.unsqueeze(0).expand(batch_size, batch_size)
        
        labels_i = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_j = labels.unsqueeze(0).expand(batch_size, batch_size)
        
        # Create mask to only compare pairs where labels differ
        valid_pairs = (labels_i != labels_j).float()
        
        # Which pairs should have a positive difference in predictions
        higher_pairs = (labels_i > labels_j).float() * valid_pairs
        
        # Calculate pairwise ranking loss with margin
        # For each pair where i should be ranked higher than j:
        # loss += max(0, margin - (logits_i - logits_j))
        losses = torch.clamp(self.margin - (logits_i - logits_j), min=0.0)
        
        # Only consider pairs where i should be ranked higher than j
        masked_losses = losses * higher_pairs
        
        # Average over all valid pairs
        num_valid = higher_pairs.sum()
        if num_valid > 0:
            loss = masked_losses.sum() / num_valid
        else:
            # Fall back to MSE if no valid pairs
            loss = F.mse_loss(logits, labels)
            
        return loss 