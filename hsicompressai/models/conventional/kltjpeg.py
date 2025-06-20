from PIL import Image
import torch
import torch.nn as nn
import torch.linalg
import numpy as np # For PSNR calculation in example usage

# --- Helper Functions for Patching ---

def extract_patches(images: torch.Tensor, patch_size: int):
    """
    Extracts non-overlapping patches from a batch of images.
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W).
        patch_size (int): Size of the square patch.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Extracted patches of shape (N_total_patches, C, patch_size, patch_size).
            - tuple: Original image shape (N, C, H, W) for reconstruction.
    """
    N, C, H, W = images.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})")

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Use unfold to extract patches efficiently
    # Resulting shape after first unfold (dim 2, height): (N, C, num_patches_h, patch_size, W)
    # Resulting shape after second unfold (dim 4, width): (N, C, num_patches_h, patch_size, num_patches_w, patch_size)
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    # Permute and reshape to get (N * num_patches_h * num_patches_w, C, patch_size, patch_size)
    # The order of permute is important: (0, 1, 2, 4, 3, 5) puts the patch_size dimensions at the end
    # Then view combines the N, num_patches_h, num_patches_w into a single batch dimension for patches.
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches = patches.view(N * num_patches_h * num_patches_w, C, patch_size, patch_size)
    
    return patches, (N, C, H, W)

def reconstruct_images_from_patches(patches: torch.Tensor, original_shape: tuple, patch_size: int):
    """
    Reconstructs images from non-overlapping patches using permute and view,
    reversing the extract_patches operation.
    Args:
        patches (torch.Tensor): Patches of shape (N_total_patches, C, patch_size, patch_size).
        original_shape (tuple): Original image shape (N, C, H, W).
        patch_size (int): Size of the square patch.

    Returns:
        torch.Tensor: Reconstructed images of shape (N, C, H, W).
    """
    N, C, H, W = original_shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # 1. Reshape patches back to their grid positions:
    # From (N_total_patches, C, patch_size, patch_size)
    # To (N, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches_reshaped = patches.view(N, num_patches_h, num_patches_w, C, patch_size, patch_size)
    
    # 2. Permute to revert the order used in extract_patches.
    # The original permute in extract_patches was (0, 1, 2, 4, 3, 5)
    # to convert from (N, C, num_patches_h, patch_size, num_patches_w, patch_size)
    # to (N, C, num_patches_h, num_patches_w, patch_size, patch_size).
    # To reverse this, we apply the inverse permutation. In this specific case,
    # the inverse permutation is the same: (0, 1, 2, 4, 3, 5)
    # It swaps the dimensions 3 and 4 back.
    patches_permuted_back = patches_reshaped.permute(0, 3, 1, 4, 2, 5).contiguous()

    # 3. Reshape to combine the patch grid dimensions with patch dimensions
    # To (N, C, H, W)
    reconstructed_images = patches_permuted_back.view(N, C, H, W)
    return reconstructed_images

# --- KLT Transform Module ---
class KLTTransform(nn.Module):
    def __init__(self, patch_size: int, num_channels: int, num_klt_components: int = None):
        """
        Initializes the KLT Transform module.
        Args:
            patch_size (int): Size of the square patch for KLT.
            num_channels (int): Number of channels in the input images.
            num_klt_components (int, optional): Number of components to keep after KLT.
                                               If None, keep all (patch_size*patch_size) components.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_dim = patch_size * patch_size

        # Default to keeping all components if not specified
        self.num_klt_components = num_klt_components if num_klt_components is not None else self.patch_dim

        if self.num_klt_components > self.patch_dim:
            raise ValueError(f"num_klt_components ({self.num_klt_components}) cannot be greater than patch_dim ({self.patch_dim})")

        # KLT basis for each channel: (num_channels, patch_dim, num_klt_components)
        self.register_buffer('klt_basis', None)
        # Mean vector for each channel: (num_channels, patch_dim)
        self.register_buffer('mean_vec', None) 

        # Store original image shape during forward for inverse reconstruction
        self._original_image_shape = None

    def fit(self, images: torch.Tensor):
        """
        Learns the KLT basis from a batch of images. This method must be called
        before using the forward pass for actual transformation.
        Args:
            images (torch.Tensor): A batch of images (N, C, H, W). 
                                   Dimensions must be divisible by patch_size.
        """
        if images.shape[1] != self.num_channels:
            raise ValueError(f"Number of channels in images ({images.shape[1]}) does not match KLTTransform init ({self.num_channels})")
        
        # Ensure image dimensions are divisible by patch_size
        H, W = images.shape[2:]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by KLT patch_size ({self.patch_size}) for fitting.")

        all_patches, _ = extract_patches(images.float(), self.patch_size)
        # all_patches shape: (N_total_patches, C, patch_size, patch_size)

        klt_basis_list = []
        mean_vec_list = []

        # Learn KLT basis independently for each channel
        for c in range(self.num_channels):
            # Extract patches for current channel and flatten them
            channel_patches = all_patches[:, c, :, :].view(-1, self.patch_dim) # (N_total_patches, patch_dim)

            # Center the data by subtracting the mean
            mean_vec = channel_patches.mean(dim=0, keepdim=True) # (1, patch_dim)
            centered_patches = channel_patches - mean_vec

            # Compute covariance matrix. torch.cov expects samples as columns, so transpose.
            # Handles cases where N_total_patches < patch_dim by giving a positive semidefinite matrix.
            covariance_matrix = torch.cov(centered_patches.T) 

            # Perform eigenvalue decomposition (eigh for symmetric matrices)
            # eigenvalues are returned in ascending order by default.
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

            # Sort eigenvectors by eigenvalues in descending order (most significant components first)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]

            # Select the top num_klt_components for the basis
            basis = eigenvectors[:, :self.num_klt_components] # (patch_dim, num_klt_components)

            klt_basis_list.append(basis)
            mean_vec_list.append(mean_vec)

        # Stack the bases for all channels: (C, patch_dim, num_klt_components)
        self.klt_basis = nn.Parameter(torch.stack(klt_basis_list, dim=0), requires_grad=False)
        # Stack the mean vectors: (C, patch_dim)
        self.mean_vec = nn.Parameter(torch.stack(mean_vec_list, dim=0).squeeze(1), requires_grad=False)

    def _transform_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Transforms flattened patches using the learned KLT basis.
        Args:
            patches (torch.Tensor): Flattened patches (N_total_patches, C, patch_dim).
        Returns:
            torch.Tensor: KLT coefficients (N_total_patches, C, num_klt_components).
        """
        N_total_patches, C, patch_dim = patches.shape
        klt_coeffs = torch.empty(N_total_patches, C, self.num_klt_components, 
                                 device=patches.device, dtype=patches.dtype)

        for c in range(C):
            # Center the patches for the current channel
            centered_patches = patches[:, c, :] - self.mean_vec[c] # (N_total_patches, patch_dim)
            # Apply KLT: coeff = centered_patch @ basis
            klt_coeffs[:, c, :] = torch.matmul(centered_patches, self.klt_basis[c])

        return klt_coeffs

    def _inverse_transform_patches(self, klt_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Inverse transforms KLT coefficients back to flattened patches.
        Args:
            klt_coeffs (torch.Tensor): KLT coefficients (N_total_patches, C, num_klt_components).
        Returns:
            torch.Tensor: Reconstructed flattened patches (N_total_patches, C, patch_dim).
        """
        N_total_patches, C, num_components = klt_coeffs.shape
        reconstructed_flat_patches = torch.empty(N_total_patches, C, self.patch_dim, 
                                                 device=klt_coeffs.device, dtype=klt_coeffs.dtype)

        for c in range(C):
            # Apply inverse KLT: patch = coeff @ basis^T
            reconstructed_centered_patch = torch.matmul(klt_coeffs[:, c, :], self.klt_basis[c].T)
            # Add back the mean
            reconstructed_flat_patches[:, c, :] = reconstructed_centered_patch + self.mean_vec[c]

        return reconstructed_flat_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies KLT (transform and reconstruction) to input images.
        This effectively applies a potentially lossy KLT filter based on `num_klt_components`.
        Args:
            x (torch.Tensor): Input images (N, C, H, W).
        Returns:
            torch.Tensor: KLT-transformed and reconstructed images (N, C, H, W).
        """
        if self.klt_basis is None:
            raise RuntimeError("KLT basis not learned. Call .fit() first with training data.")

        # Store original image shape for reconstruction
        _, _, H, W = x.shape
        self._original_image_shape = (x.shape[0], x.shape[1], H, W)

        # 1. Extract patches
        patches, original_shape_info = extract_patches(x.float(), self.patch_size)
        # 2. Flatten patches: (N_total_patches, C, patch_dim)
        flat_patches = patches.view(-1, patches.shape[1], self.patch_dim)

        # 3. Transform to KLT coefficients
        klt_coeffs = self._transform_patches(flat_patches)

        # 4. Inverse transform from KLT coefficients (with potential dimensionality reduction)
        reconstructed_flat_patches = self._inverse_transform_patches(klt_coeffs)

        # 5. Reshape flat patches back to (N_total_patches, C, patch_size, patch_size)
        reconstructed_patches = reconstructed_flat_patches.view(-1, self.num_channels, 
                                                               self.patch_size, self.patch_size)

        # 6. Reconstruct full images from patches
        reconstructed_images = reconstruct_images_from_patches(reconstructed_patches, original_shape_info, self.patch_size)

        return reconstructed_images

# --- Simplified Haar Wavelet Transform Module ---
class WaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Performs a single-level 2D Haar DWT.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            tuple: A tuple containing (LL, LH, HL, HH) subbands,
                   each of shape (N, C, H/2, W/2).
        """
        N, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("Image dimensions must be even for Haar DWT.")

        # Step 1: Horizontal transformation (along W dimension)
        # Split into even and odd columns
        x_even_cols = x[..., ::2]
        x_odd_cols = x[..., 1::2]

        # Horizontal Low-Pass (L) and High-Pass (H) bands
        # (N, C, H, W/2)
        L_h = (x_even_cols + x_odd_cols) / 2.0
        H_h = (x_even_cols - x_odd_cols) / 2.0

        # Step 2: Vertical transformation (along H dimension)
        # Combine L_h and H_h to process vertically
        # Stack them along a new dimension to treat as a single tensor for vertical processing
        # (N, C, H, W/2, 2) where last dim distinguishes L_h and H_h
        temp_bands = torch.stack((L_h, H_h), dim=-1)

        # Split into even and odd rows for vertical transformation
        # (N, C, H/2, W/2, 2)
        temp_bands_even_rows = temp_bands[:, :, ::2, :, :]
        temp_bands_odd_rows = temp_bands[:, :, 1::2, :, :]

        # Apply vertical Haar transform
        # LL (Low-Low): Vertical L-pass on L_h
        LL = (temp_bands_even_rows[..., 0] + temp_bands_odd_rows[..., 0]) / 2.0
        # LH (Low-High): Vertical H-pass on L_h
        LH = (temp_bands_even_rows[..., 0] - temp_bands_odd_rows[..., 0]) / 2.0
        # HL (High-Low): Vertical L-pass on H_h
        HL = (temp_bands_even_rows[..., 1] + temp_bands_odd_rows[..., 1]) / 2.0
        # HH (High-High): Vertical H-pass on H_h
        HH = (temp_bands_even_rows[..., 1] - temp_bands_odd_rows[..., 1]) / 2.0

        # Each subband has shape (N, C, H/2, W/2)
        return LL, LH, HL, HH

    def inverse(self, subbands: tuple) -> torch.Tensor:
        """
        Performs a single-level 2D Haar IWT.
        Args:
            subbands (tuple): Tuple of (LL, LH, HL, HH) subbands, 
                              each of shape (N, C, H/2, W/2).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (N, C, H, W).
        """
        LL, LH, HL, HH = subbands
        N, C, H_half, W_half = LL.shape

        # Step 1: Inverse Vertical Transformation
        # Reconstruct intermediate rows for L_h and H_h parts
        # L_h_even_rows = LL + LH, L_h_odd_rows = LL - LH
        L_h_reconstructed_even = LL + LH
        L_h_reconstructed_odd = LL - LH
        
        # H_h_even_rows = HL + HH, H_h_odd_rows = HL - HH
        H_h_reconstructed_even = HL + HH
        H_h_reconstructed_odd = HL - HH

        # Interleave rows to form L_h_reconstructed and H_h_reconstructed
        # (N, C, H_half * 2, W_half)
        L_h_reconstructed = torch.empty((N, C, H_half * 2, W_half), device=LL.device, dtype=LL.dtype)
        L_h_reconstructed[:, :, ::2, :] = L_h_reconstructed_even
        L_h_reconstructed[:, :, 1::2, :] = L_h_reconstructed_odd

        H_h_reconstructed = torch.empty((N, C, H_half * 2, W_half), device=HL.device, dtype=HL.dtype)
        H_h_reconstructed[:, :, ::2, :] = H_h_reconstructed_even
        H_h_reconstructed[:, :, 1::2, :] = H_h_reconstructed_odd

        # Step 2: Inverse Horizontal Transformation
        # Reconstruct original image by interleaving columns
        # original_even_cols = L_h_reconstructed + H_h_reconstructed
        # original_odd_cols = L_h_reconstructed - H_h_reconstructed

        reconstructed_image = torch.empty((N, C, H_half * 2, W_half * 2), device=LL.device, dtype=LL.dtype)
        reconstructed_image[:, :, :, ::2] = L_h_reconstructed + H_h_reconstructed
        reconstructed_image[:, :, :, 1::2] = L_h_reconstructed - H_h_reconstructed

        return reconstructed_image

# --- Quantization Module ---
class Quantizer(nn.Module):
    def __init__(self, quant_step: float):
        """
        Initializes the uniform quantizer.
        Args:
            quant_step (float): The quantization step size.
        """
        super().__init__()
        self.quant_step = float(quant_step)
        if self.quant_step <= 0:
            raise ValueError("Quantization step must be positive.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies uniform quantization to the input tensor.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Quantized tensor.
        """
        return torch.round(x / self.quant_step) * self.quant_step

    def inverse(self, x_quantized: torch.Tensor) -> torch.Tensor:
        """
        Dequantization (identity for uniform quantization with no additional info).
        Args:
            x_quantized (torch.Tensor): Quantized tensor.
        Returns:
            torch.Tensor: Dequantized tensor (same as input for uniform quantizer).
        """
        return x_quantized

# --- KLT + Simplified JPEG2000 Hybrid Module ---
class KLT_JPEG2000_Hybrid(nn.Module):
    def __init__(self, patch_size: int, num_channels: int, quant_step: float, num_klt_components: int = None):
        """
        Initializes the KLT + Simplified JPEG2000 hybrid compression module.
        Args:
            patch_size (int): Size of the square patch for KLT.
            num_channels (int): Number of channels in the input images.
            quant_step (float): Quantization step size for wavelet coefficients.
            num_klt_components (int, optional): Number of components to keep after KLT.
                                               If None, keep all (patch_size*patch_size) components.
        """
        super().__init__()
        self.klt_transform = KLTTransform(patch_size, num_channels, num_klt_components)
        self.wavelet_transform = WaveletTransform()
        self.quantizer = Quantizer(quant_step)

        self.patch_size = patch_size
        self.num_channels = num_channels
        self.quant_step = quant_step
        self.num_klt_components = num_klt_components

    def fit_klt_basis(self, images: torch.Tensor):
        """
        Learns the KLT basis from a batch of images. This should be called once
        before performing compression/decompression.
        Args:
            images (torch.Tensor): A batch of images (N, C, H, W) to learn KLT from.
                                   Dimensions must be divisible by patch_size.
        """
        print("Fitting KLT basis for the hybrid module...")
        self.klt_transform.fit(images)
        print("KLT basis fitted.")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Compresses an image using KLT followed by DWT and Quantization.
        Args:
            x (torch.Tensor): Input images of shape (N, C, H, W).
                              Dimensions must be divisible by KLT patch_size and be even for DWT.
        Returns:
            dict: A dictionary containing the quantized wavelet subbands:
                  {'LL': qLL, 'LH': qLH, 'HL': qHL, 'HH': qHH}.
        """
        N, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image H ({H}) and W ({W}) must be divisible by KLT patch_size ({self.patch_size}).")
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"Image H ({H}) and W ({W}) must be even for DWT (single level Haar).")

        # Step 1: Apply KLT (transform and reconstruct, potentially with loss)
        # The output `klt_processed_image` is an image where each patch has been KLT-transformed
        # and then reconstructed, potentially with reduced dimensionality.
        klt_processed_image = self.klt_transform(x.float())

        # Step 2: Apply Wavelet Transform (DWT) on the KLT-processed image
        LL, LH, HL, HH = self.wavelet_transform(klt_processed_image)

        # Step 3: Quantize the wavelet coefficients
        qLL = self.quantizer(LL)
        qLH = self.quantizer(LH)
        qHL = self.quantizer(HL)
        qHH = self.quantizer(HH)
        
        # In a real compression system, these quantized coefficients would be
        # fed into an entropy coder (e.g., EBCOT for JPEG2000).
        # Here, we return them directly as the 'compressed' representation.
        return {'LL': qLL, 'LH': qLH, 'HL': qHL, 'HH': qHH}

    def inverse(self, subbands_quantized: dict) -> torch.Tensor:
        """
        Decompresses the quantized wavelet subbands to reconstruct the image.
        Args:
            subbands_quantized (dict): Dictionary containing quantized subbands
                                       {'LL': qLL, 'LH': qLH, 'HL': qHL, 'HH': qHH}.
        Returns:
            torch.Tensor: Reconstructed image of shape (N, C, H, W).
        """
        # Step 1: Dequantize the wavelet coefficients
        LL = self.quantizer.inverse(subbands_quantized['LL'])
        LH = self.quantizer.inverse(subbands_quantized['LH'])
        HL = self.quantizer.inverse(subbands_quantized['HL'])
        HH = self.quantizer.inverse(subbands_quantized['HH'])

        # Step 2: Apply Inverse Wavelet Transform (IWT)
        dwt_reconstructed_image = self.wavelet_transform.inverse((LL, LH, HL, HH))

        # Note: We do NOT apply inverse KLT here in the inverse function for the hybrid.
        # The KLT transformation (and potential lossy reconstruction) was already applied
        # in the `forward` pass by `self.klt_transform(x)`. The `dwt_reconstructed_image`
        # is the KLT-processed image, now dequantized and inverse DWT'd.
        # This will be an approximation of the original input image, due to KLT dimensionality
        # reduction and quantization.

        return dwt_reconstructed_image

if __name__ == "__main__":
    # Create a dummy image batch (e.g., 2 images, 3 channels, 32x32 pixels)
    # Ensure dimensions are divisible by patch_size and are even for DWT
    image = torch.from_numpy(np.array(Image.open('/home/joshua/Documents/phd_university/code/HSICompressAI/assests/dog.png').convert('RGB')))

    batch_size = 1
    height, width, channels = image.shape
    image = image.clamp(0, 255).to(torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    image = image[:,:,:4080,:6120]

    patch_size = 8
    quant_step = 5.0
    num_klt_components = 32 # Keep 32 components out of 8*8=64 possible. This makes KLT lossy.

    print(f"Original image shape: {image.shape}")
    klt_transform = KLTTransform(patch_size=patch_size, num_channels=channels, num_klt_components=num_klt_components)
    klt_transform.fit(image)
    klt_processed_image = klt_transform(image.float())

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image (Batch 0, Channel 0)")
        plt.imshow(image[0, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image (Batch 0, Channel 0)")
        plt.imshow(klt_processed_image[0, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")

    # # Initialize the hybrid module
    # hybrid_compressor = KLT_JPEG2000_Hybrid(
    #     patch_size=patch_size,
    #     num_channels=channels,
    #     quant_step=quant_step,
    #     num_klt_components=num_klt_components
    # )

    # # 1. Fit the KLT basis (essential training step)
    # # This uses the image to learn the KLT transform.

    # # 2. Compress the images
    # print("\nCompressing images...")
    # compressed_subbands = hybrid_compressor(image)
    
    # print("Compressed subband shapes:")
    # for key, value in compressed_subbands.items():
    #     print(f"  {key}: {value.shape}")
    # print(f"Example LL subband min: {compressed_subbands['LL'].min()}, max: {compressed_subbands['LL'].max()}")

    # # 3. Decompress the images
    # print("\nDecompressing images...")
    # reconstructed_images = hybrid_compressor.inverse(compressed_subbands)
    
    # print(f"Reconstructed image shape: {reconstructed_images.shape}")
    # print(f"Reconstructed image min: {reconstructed_images.min()}, max: {reconstructed_images.max()}")

    # # 4. Calculate PSNR (Peak Signal-to-Noise Ratio) to evaluate compression quality
    # # For PSNR, values typically in [0, 255] range.
    # # Convert tensors to CPU numpy for easier handling
    # original_np = image.cpu().numpy()
    # reconstructed_np = reconstructed_images.cpu().numpy()

    # # Ensure the reconstructed image is within the valid range for PSNR calculation
    # reconstructed_np = np.clip(reconstructed_np, 0, 255)

    # def calculate_psnr(img1, img2, max_pixel_value=255.0):
    #     mse = np.mean((img1 - img2) ** 2)
    #     if mse == 0:
    #         return float('inf')
    #     psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    #     return psnr

    # import numpy as np # Import numpy for PSNR calculation
    # psnr_value = calculate_psnr(original_np, reconstructed_np)
    # print(f"\nPSNR between original and reconstructed image: {psnr_value:.2f} dB")

    # # Visualize (optional, requires matplotlib)
    # try:
    #     import matplotlib.pyplot as plt

    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.title("Original Image (Batch 0, Channel 0)")
    #     plt.imshow(image[0, 0].cpu().numpy(), cmap='gray')
    #     plt.axis('off')

    #     plt.subplot(1, 2, 2)
    #     plt.title("Reconstructed Image (Batch 0, Channel 0)")
    #     plt.imshow(reconstructed_images[0, 0].cpu().numpy(), cmap='gray')
    #     plt.axis('off')
    #     plt.show()

    # except ImportError:
    #     print("\nMatplotlib not found. Skipping visualization.")
