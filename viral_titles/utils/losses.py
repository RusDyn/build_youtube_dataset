"""
Custom loss functions for viral titles training.
"""
import torch
import torch.nn.functional as F

class PairwiseRankingLoss:
    """
    Enhanced implementation of a pairwise ranking loss function optimized for viral score prediction.
    Features:
    - Adaptive margin based on score differences
    - Sample weighting to focus on more important pairs
    - Option to use either hinge loss (margin-based) or soft ranking loss
    """
    def __init__(self, base_margin=0.1, adaptive_margin=True, loss_type="hinge", 
                 focus_on_boundary=True, temperature=1.0):
        """
        Initialize the pairwise ranking loss function.
        
        Args:
            base_margin: Base margin for hinge loss (default: 0.1)
            adaptive_margin: Whether to scale margin by label difference (default: True)
            loss_type: Type of loss to use - "hinge" or "soft" (default: "hinge")
            focus_on_boundary: Whether to give higher weight to pairs near decision boundary (default: True)
            temperature: Temperature parameter for soft loss (default: 1.0)
        """
        self.base_margin = base_margin
        self.adaptive_margin = adaptive_margin
        self.loss_type = loss_type
        self.focus_on_boundary = focus_on_boundary
        self.temperature = temperature
        
    def __call__(self, logits, labels):
        """
        Compute enhanced pairwise ranking loss.
        
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
        
        # Create mask for valid pairs (different labels)
        label_diff = labels_i - labels_j
        valid_pairs = (label_diff != 0).float()
        
        # Which pairs should have a positive difference in predictions
        # i.e., label_i > label_j should imply logit_i > logit_j
        higher_pairs = (label_diff > 0).float() * valid_pairs
        
        # Calculate adaptive margin based on label differences if enabled
        if self.adaptive_margin:
            # Scale margin by label difference, with minimum of base_margin
            # This makes the loss require larger margins for pairs with bigger label differences
            margin = torch.clamp(label_diff.abs() * 2.0, min=self.base_margin) * higher_pairs
        else:
            margin = self.base_margin * higher_pairs
            
        # Compute prediction differences
        pred_diff = logits_i - logits_j
        
        # Optional: Focus on boundary cases by weighting
        if self.focus_on_boundary:
            # Higher weight for pairs that are incorrectly ranked or close to margin
            # i.e., focus on pairs where pred_diff < margin for higher_pairs=1
            boundary_weight = 1.0 + torch.exp(-10.0 * (pred_diff - 0.5 * margin) / margin)
            boundary_weight = torch.clamp(boundary_weight, max=5.0) * higher_pairs
        else:
            boundary_weight = higher_pairs
        
        # Apply the chosen loss type
        if self.loss_type == "hinge":
            # Hinge loss for ranking: max(0, margin - pred_diff) when label_i > label_j
            losses = torch.clamp(margin - pred_diff, min=0.0)
            weighted_losses = losses * boundary_weight
        else:  # soft ranking loss
            # Soft ranking loss using sigmoid of negative difference scaled by temperature
            # For pairs where i should be higher than j
            scale = 1.0 / self.temperature
            sigmoids = torch.sigmoid(-scale * pred_diff)
            weighted_losses = sigmoids * boundary_weight
            
        # Average over all valid pairs
        num_valid = higher_pairs.sum()
        if num_valid > 0:
            loss = weighted_losses.sum() / num_valid
        else:
            # Fall back to MSE if no valid pairs
            loss = F.mse_loss(logits, labels)
            
        return loss 