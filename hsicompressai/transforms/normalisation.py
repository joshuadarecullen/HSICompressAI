import torch

def range_adaptive_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Applies range-adaptive normalization to a tensor, per channel.

    Args:
        x (torch.Tensor): The input tensor with shape (Batch_Size, Channels, Height, Width)
                          or (Channels, Height, Width) if it's a single image.

    Returns:
        torch.Tensor: The normalized tensor.
        (torch.Tensor, torch.Tensor): The MIN_b and MAX_b values (side information)
                                      for each channel.
    """
    # Ensure x is float for division
    x = x.float()

    # Calculate MIN_b and MAX_b for each band (channel)
    # We need to find the min/max across Height and Width dimensions for each channel.
    # If the input is (B, C, H, W), we want min/max over (H, W) for each (B, C) pair.
    # To get MIN_b and MAX_b for each 'b' (channel) independently across all batches and spatial dimensions,
    # we first flatten the spatial dimensions.
    # Then we find the min/max along the flattened spatial dimensions for each channel.

    # Option 1: Calculate MIN_b and MAX_b per-channel across all images in the batch
    # This assumes MIN_b and MAX_b are global statistics for the dataset,
    # or at least for the entire batch.
    # This is often how it's handled in image processing for consistency.

    # Reshape to (Batch_Size * Channels, Height * Width) to calculate min/max per channel efficiently
    # No, this is incorrect. We need min/max per *channel*, not per (batch, channel).
    # If we assume MIN_b and MAX_b are per-channel statistics *across the entire dataset*,
    # then you would pre-compute these values.
    # However, the description "MIN_b and MAX_b for each of the k bands need to be stored as 'side information'"
    # implies they are calculated for the *current input* (e.g., current image or current batch).

    # Let's assume the most common interpretation in deep learning:
    # MIN_b and MAX_b are calculated *per channel* over the *entire image/batch*.
    # So if your tensor is (B, C, H, W), we calculate min/max for each of the C channels
    # across B, H, W.

    # To get min/max per channel:
    # We need to reduce over batch (0), height (2), and width (3) dimensions,
    # keeping the channel dimension (1).
    
    # Keepdim=True ensures the output shape maintains the dimensions for broadcasting.
    # e.g., (B, C, H, W) -> (1, C, 1, 1) after min/max
    min_b = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=0, keepdim=True)[0]
    max_b = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
    
    # The above assumes a 4D tensor (Batch, Channel, Height, Width).
    # If your input can also be a single image (Channel, Height, Width),
    # you might need to adjust the dim argument, or handle it with an unsqueeze.
    
    # A more robust way to handle both (B, C, H, W) and (C, H, W) is to
    # calculate min/max over all dimensions except the channel dimension.
    # Assuming channel is always at index 1 for (B, C, H, W) or index 0 for (C, H, W).
    
    # To make it work generally for NCHW or CHW, find the channel dimension dynamically:
    if x.dim() == 4: # (Batch, Channels, Height, Width)
        # Calculate min/max over B, H, W for each C
        min_b_all_dims = x.min(dim=0, keepdim=True)[0] # Min over batch
        min_b_all_dims = min_b_all_dims.min(dim=-1, keepdim=True)[0] # Min over width
        min_b = min_b_all_dims.min(dim=-2, keepdim=True)[0] # Min over height
        
        max_b_all_dims = x.max(dim=0, keepdim=True)[0] # Max over batch
        max_b_all_dims = max_b_all_dims.max(dim=-1, keepdim=True)[0] # Max over width
        max_b = max_b_all_dims.max(dim=-2, keepdim=True)[0] # Max over height

        # The shape of min_b and max_b will be (1, C, 1, 1), which is perfect for broadcasting.

    elif x.dim() == 3: # (Channels, Height, Width) - single image
        # Calculate min/max over H, W for each C
        min_b = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_b = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        # The shape of min_b and max_b will be (C, 1, 1), perfect for broadcasting.
    else:
        raise ValueError("Input tensor must be 3D (CHW) or 4D (NCHW).")

    # Apply the normalization formula: x'_{b,l,i} = (x_{b,l,i} - MIN_b) / (MAX_b - MIN_b + 1)
    
    # Denominator to prevent division by zero when MIN_b == MAX_b
    denominator = (max_b - min_b + 1).clamp(min=1e-8) # Add a small epsilon if MAX_b - MIN_b is 0

    x_normalized = (x - min_b) / denominator

    return x_normalized, min_b.squeeze(), max_b.squeeze() # Squeeze to remove singleton dimensions for side info


# Example Usage:

# 1. For a single image (Channels, Height, Width)
image = torch.randint(0, 65536, (3, 256, 256)).float() # Example 16-bit image (0-65535)
normalized_image, min_vals, max_vals = range_adaptive_normalization(image)

print(f"Original Image Shape: {image.shape}")
print(f"Normalized Image Shape: {normalized_image.shape}")
print(f"Min values per channel: {min_vals}")
print(f"Max values per channel: {max_vals}")
print(f"Normalized image min: {normalized_image.min()}, max: {normalized_image.max()}")
print("-" * 30)

# 2. For a batch of images (Batch_Size, Channels, Height, Width)
batch_images = torch.randint(0, 65536, (4, 3, 128, 128)).float()
normalized_batch, batch_min_vals, batch_max_vals = range_adaptive_normalization(batch_images)

print(f"Original Batch Shape: {batch_images.shape}")
print(f"Normalized Batch Shape: {normalized_batch.shape}")
print(f"Min values per channel (across batch): {batch_min_vals}")
print(f"Max values per channel (across batch): {batch_max_vals}")
print(f"Normalized batch min: {normalized_batch.min()}, max: {normalized_batch.max()}")
print("-" * 30)

# Special case: All values in a band are the same (MIN_b == MAX_b)
# This will test the '+1' and clamp for division by zero.
test_tensor_same_values = torch.tensor([
    [[[10.0, 10.0], [10.0, 10.0]], [[20.0, 20.0], [20.0, 20.0]]],
    [[[10.0, 10.0], [10.0, 10.0]], [[20.0, 20.0], [20.0, 20.0]]]
]).float() # Shape (B=2, C=2, H=2, W=2)

normalized_test, test_min, test_max = range_adaptive_normalization(test_tensor_same_values)
print(f"Test tensor (same values per channel) normalized: \n{normalized_test}")
print(f"Test min values: {test_min}")
print(f"Test max values: {test_max}")
print(f"Normalized test min: {normalized_test.min()}, max: {normalized_test.max()}")
