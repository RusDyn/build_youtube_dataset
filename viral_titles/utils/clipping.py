"""
Utilities for clipping and scaling prediction values.
"""
import numpy as np

def soft_clip(x, margin=0.1):
    """
    Apply soft clipping to values to keep them in the [0, 1] range
    while preserving more of the distribution.
    
    Args:
        x: Input values (numpy array or scalar)
        margin: Controls the "softness" of the clipping
               (smaller values = sharper transition)
               
    Returns:
        Clipped values in [0, 1] range
    """
    # More gradual clipping using tanh with adjusted scale
    # Increase the scale factor to make the transition more gradual
    scale_factor = max(0.2, margin)  # Ensure scale is at least 0.2 for gradual transition
    return 0.5 + 0.5 * np.tanh((x - 0.5) / scale_factor)

def linear_clip(x, margin=0.05):
    """
    Alternative clipping function using linear interpolation with margins
    
    Args:
        x: Input values (numpy array or scalar)
        margin: Margin at edges
        
    Returns:
        Clipped values in [0, 1] range
    """
    # Ensure x is a numpy array
    x = np.asarray(x)
    # Linear mapping from [0+margin, 1-margin] to [0, 1]
    return np.clip((x - margin) / (1 - 2 * margin), 0, 1) 