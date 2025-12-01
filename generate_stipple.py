"""
Standalone script to generate blue noise stippling from an image.
This will create:
1. A stippled version of the image
2. A progressive stippling GIF animation
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# ============================================================================
# Load Image
# ============================================================================
print("Loading image...")
img_path = 'florida_beach_picture.jpg'
original_img = Image.open(img_path)

# Convert to grayscale if needed
if original_img.mode != 'L':
    original_img = original_img.convert('L')

# Convert to numpy array and normalize to [0, 1]
img_array = np.array(original_img, dtype=np.float32) / 255.0
print(f"Image loaded: {img_array.shape[0]}x{img_array.shape[1]} pixels")

# ============================================================================
# Importance Map Function
# ============================================================================
def compute_importance(
    gray_img: np.ndarray,
    extreme_downweight: float = 0.5,
    extreme_threshold_low: float = 0.4,
    extreme_threshold_high: float = 0.8,
    extreme_sigma: float = 0.1,
    mid_tone_boost: float = 0.4,
    mid_tone_sigma: float = 0.2,
):
    """Compute importance map for stippling."""
    I = np.clip(gray_img, 0.0, 1.0)
    I_inverted = 1.0 - I
    
    # Dark mask
    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask = dark_mask / dark_mask.max()
    
    # Light mask
    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask = light_mask / light_mask.max()
    
    extreme_mask = np.maximum(dark_mask, light_mask)
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
    
    # Mid-tone boost
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    
    importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
    
    # Normalize
    m, M = importance.min(), importance.max()
    if M > m: 
        importance = (importance - m) / (M - m)
    return importance

# ============================================================================
# Stippling Functions
# ============================================================================
def toroidal_gaussian_kernel(h: int, w: int, sigma: float):
    """Create a periodic (toroidal) 2D Gaussian kernel."""
    y = np.arange(h)
    x = np.arange(w)
    dy = np.minimum(y, h - y)[:, None]
    dx = np.minimum(x, w - x)[None, :]
    kern = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    s = kern.sum()
    if s > 0:
        kern /= s
    return kern

def void_and_cluster(
    input_img: np.ndarray,
    percentage: float = 0.08,
    sigma: float = 0.9,
    content_bias: float = 0.9,
    importance_img: np.ndarray | None = None,
    noise_scale_factor: float = 0.1,
):
    """Generate blue noise stippling pattern."""
    I = np.clip(input_img, 0.0, 1.0)
    h, w = I.shape

    # Compute or use provided importance map
    if importance_img is None:
        importance = compute_importance(I)
    else:
        importance = np.clip(importance_img, 0.0, 1.0)

    # Create toroidal Gaussian kernel
    kernel = toroidal_gaussian_kernel(h, w, sigma)

    # Initialize energy field
    energy_current = -importance * content_bias

    # Stipple buffer
    final_stipple = np.ones_like(I)
    samples = []

    def energy_splat(y, x):
        return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)

    # Number of points to select
    num_points = int(I.size * percentage)
    print(f"Generating {num_points} stipple points...")

    # Choose first point near center
    cy, cx = h // 2, w // 2
    r = min(20, h // 10, w // 10)
    ys = slice(max(0, cy - r), min(h, cy + r))
    xs = slice(max(0, cx - r), min(w, cx + r))
    region = energy_current[ys, xs]
    flat = np.argmin(region)
    y0 = flat // (region.shape[1]) + (cy - r)
    x0 = flat % (region.shape[1]) + (cx - r)

    # Place first point
    energy_current = energy_current + energy_splat(y0, x0)
    energy_current[y0, x0] = np.inf
    samples.append((y0, x0, I[y0, x0]))
    final_stipple[y0, x0] = 0.0

    # Iteratively place remaining points
    for i in range(1, num_points):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{num_points} points ({100*(i+1)/num_points:.1f}%)")
        
        exploration = 1.0 - (i / num_points) * 0.5
        noise = np.random.normal(0.0, noise_scale_factor * content_bias * exploration, size=energy_current.shape)
        energy_with_noise = energy_current + noise

        pos_flat = np.argmin(energy_with_noise)
        y = pos_flat // w
        x = pos_flat % w

        energy_current = energy_current + energy_splat(y, x)
        energy_current[y, x] = np.inf

        samples.append((y, x, I[y, x]))
        final_stipple[y, x] = 0.0

    return final_stipple, np.array(samples)

# ============================================================================
# Prepare Image
# ============================================================================
print("\nPreparing image...")
max_size = 512
if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
    scale = max_size / max(img_array.shape[0], img_array.shape[1])
    new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
    img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
    if img_resized_pil.mode != 'L':
        img_resized_pil = img_resized_pil.convert('L')
    img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
    print(f"Resized image from {img_array.shape} to {img_resized.shape}")
else:
    img_resized = img_array.copy()

# Ensure 2D grayscale
if len(img_resized.shape) > 2:
    img_resized = img_resized[:, :, 0]

print(f"Final image shape: {img_resized.shape}")

# Compute importance map
print("Computing importance map...")
importance_map = compute_importance(
    img_resized,
    extreme_downweight=0.5,
    extreme_threshold_low=0.2,
    extreme_threshold_high=0.8,
    extreme_sigma=0.1
)

# ============================================================================
# Generate Stippling
# ============================================================================
print("\nGenerating blue noise stippling pattern...")
stipple_pattern, samples = void_and_cluster(
    img_resized,
    percentage=0.08,
    sigma=0.9,
    content_bias=0.9,
    importance_img=importance_map,
    noise_scale_factor=0.1
)

print(f"\nGenerated {len(samples)} stipple points")
print(f"Stipple pattern shape: {stipple_pattern.shape}")

# ============================================================================
# Save Stippled Image
# ============================================================================
print("\nSaving stippled image...")
stipple_img = Image.fromarray((stipple_pattern * 255).astype(np.uint8), mode='L')
stipple_img.save('stippled_image.png')
print("Saved: stippled_image.png")

# ============================================================================
# Create Comparison Figure
# ============================================================================
print("Creating comparison figure...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_resized, cmap='gray', vmin=0, vmax=1)
axes[0].axis('off')
axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)

axes[1].imshow(importance_map, cmap='gray', vmin=0, vmax=1)
axes[1].axis('off')
axes[1].set_title('Importance Map', fontsize=14, fontweight='bold', pad=10)

axes[2].imshow(stipple_pattern, cmap='gray', vmin=0, vmax=1)
axes[2].axis('off')
axes[2].set_title('Blue Noise Stippling', fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('stippling_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: stippling_comparison.png")
plt.close()

# ============================================================================
# Create Progressive GIF
# ============================================================================
print("\nCreating progressive stippling GIF...")
print(f"Using {len(samples)} points")
print(f"Image shape: {img_resized.shape}")

frame_increment = 100
frames = []
point_counts = []

# Start with white background
h, w = img_resized.shape
progressive_stipple = np.ones_like(img_resized)

# Add first point
if len(samples) > 0:
    y0, x0 = int(samples[0, 0]), int(samples[0, 1])
    progressive_stipple[y0, x0] = 0.0
    frames.append(progressive_stipple.copy())
    point_counts.append(1)

# Add remaining points
for i in range(1, len(samples)):
    y, x = int(samples[i, 0]), int(samples[i, 1])
    progressive_stipple[y, x] = 0.0
    
    if (i + 1) % frame_increment == 0 or i == len(samples) - 1:
        frames.append(progressive_stipple.copy())
        point_counts.append(i + 1)

print(f"Generated {len(frames)} frames")
print(f"Point counts: {point_counts}")

# Create GIF
print("Saving GIF...")
fig, ax = plt.subplots(figsize=(7, 5))
ax.axis('off')

writer = PillowWriter(fps=2)
gif_path = 'progressive_stippling.gif'

with writer.saving(fig, gif_path, dpi=100):
    for i in range(len(frames)):
        ax.clear()
        ax.axis('off')
        ax.imshow(frames[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Progressive Stippling: {point_counts[i]} points', 
                     fontsize=14, fontweight='bold', pad=10)
        writer.grab_frame()

plt.close()

print(f"Saved: {gif_path}")
print("\n✅ All done! Generated files:")
print("  - stippled_image.png")
print("  - stippling_comparison.png")
print("  - progressive_stippling.gif")

