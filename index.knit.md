---
title: "Selection Bias & Missing Data Challenge - Part 1"
subtitle: "Blue Noise Stippling: Creating Art from Data"
format:
  html: default
  pdf: default
execute:
  echo: true
  eval: true
python:
  reticulate: false
---

# 🎨 Selection Bias & Missing Data Challenge - Part 1

::: {.callout-important}
## 📊 Challenge Requirements

**Your Task:** Reproduce the blue noise stippling process demonstrated below to create:
1. A stippled version of your chosen image
2. A progressive stippling GIF animation
3. Post both to a GitHub Pages site with appropriate captions and a brief explanation

**Part 2 Preview:** On November 18th, we'll tackle Part 2 of this challenge, where you'll create a statistical meme about selection bias and missing data using your stippled images.
:::

## The Problem: Can Algorithms Create Art?

**Core Question:** How can we convert a photograph into an aesthetically pleasing pattern of dots that preserves the visual information of the original image?

**The Challenge:** Blue noise stippling is a technique that converts images into patterns of dots (stipples) using algorithms that balance visual accuracy with spatial distribution. This challenge asks you to implement a modified "void and cluster" algorithm that combines importance sampling with blue noise distribution properties to create stippling patterns that are both visually accurate and spatially well-distributed.

**Our Approach:** We'll use a modified void-and-cluster algorithm that:
1. Creates an importance map identifying visually important regions
2. Uses a toroidal (periodic) Gaussian kernel for repulsion to ensure blue noise properties
3. Iteratively selects points with minimum energy
4. Balances image content importance with blue noise spatial distribution

::: {.callout-warning}
## ⚠️ AI Partnership Required

This challenge pushes boundaries intentionally. You'll tackle problems that normally require weeks of study, but with Cursor AI as your partner (and your brain keeping it honest), you can accomplish more than you thought possible.

**The new reality:** The four stages of competence are Ignorance → Awareness → Learning → Mastery. AI lets us produce Mastery-level work while operating primarily in the Awareness stage. I focus on awareness training, you leverage AI for execution, and together we create outputs that used to require years of dedicated study.
:::

## Introduction to Blue Noise Stippling

Blue noise stippling is a technique for converting images into a pattern of dots (stipples) that preserves the visual information of the original image while creating an aesthetically pleasing, evenly distributed pattern. This method follows the approach described by [Bart Wronski](https://bartwronski.com/2022/08/31/progressive-image-stippling-and-greedy-blue-noise-importance-sampling/).

The method uses a modified "void and cluster" algorithm that combines importance sampling with blue noise distribution properties to create stippling patterns that are both visually accurate and spatially well-distributed. This version uses **smooth extreme downweighting** that selectively downweights very dark and very light tones while preserving mid-tones, creating a more balanced distribution of stipples across the image.

## Loading the Original Image

First, let's load an image that we'll convert to a blue noise stippling pattern. You can use any image you'd like, but we'll demonstrate with the provided example.

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load the image
img_path = 'florida_beach_picture.jpg'
original_img = Image.open(img_path)

# Convert to grayscale if needed
if original_img.mode != 'L':
    original_img = original_img.convert('L')

# Convert to numpy array and normalize to [0, 1]
img_array = np.array(original_img, dtype=np.float32) / 255.0

# Display the original image
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.imshow(img_array, cmap='gray', vmin=0, vmax=1)
ax.axis('off')
ax.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
# plt.show() removed - Quarto will handle figure display

print(f"Image shape: {img_array.shape}")
print(f"Image size: {img_array.shape[0]}x{img_array.shape[1]} pixels")
```


::: {.cell-output .cell-output-stdout}

```
Image shape: (1060, 1600)
Image size: 1060x1600 pixels
```


:::
:::


### R


::: {.cell}

```{.r .cell-code}
library(imager)
library(ggplot2)

# Load the image
img_path <- 'florida_beach_picture.jpg'
original_img <- load.image(img_path)

# Convert to grayscale if needed
if(dim(original_img)[4] == 3) {
  original_img <- grayscale(original_img)
}

# Display the original image
plot(original_img, axes = FALSE, main = "Original Image")

cat("Image dimensions:", dim(original_img)[1], "x", dim(original_img)[2], "pixels\n")
```
:::


:::

## Importance Mapping

Before applying the stippling algorithm, we create an **importance map** that identifies which regions of the image should receive more stipples. The importance map is computed by:

- **Brightness inversion**: The image brightness is inverted so that dark areas receive higher importance and thus more dots, while light areas receive fewer dots
- **Extreme tone downweighting**: Smooth Gaussian functions downweight tones below 0.2 (very dark) and above 0.8 (very light), creating a gradual transition that preserves mid-tones
- **Mid-tone boost**: A smooth Gaussian function centered on mid-tones provides a gradual increase in importance for mid-tone regions, ensuring they receive appropriate stippling density
- **Selective and effective**: This approach ensures that stipples are distributed appropriately (more dots in dark areas and mid-tones, fewer in extreme dark/light areas) while maintaining good spatial distribution

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np

def compute_importance(
    gray_img: np.ndarray,
    extreme_downweight: float = 0.5,
    extreme_threshold_low: float = 0.4,
    extreme_threshold_high: float = 0.8,
    extreme_sigma: float = 0.1,
    mid_tone_boost: float = 0.4,
    mid_tone_sigma: float = 0.2,
):
    """
    Importance map computation that downweights extreme tones (very dark and very light)
    using smooth functions, while boosting mid-tones.
    
    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image in [0, 1]
    extreme_downweight : float
        Strength of downweighting for extreme tones (0.0 = no downweighting, 1.0 = maximum downweighting)
    extreme_threshold_low : float
        Threshold below which tones are considered "very dark" and get downweighted
    extreme_threshold_high : float
        Threshold above which tones are considered "very light" and get downweighted
    extreme_sigma : float
        Width of the smooth transition for extreme downweighting (smaller = sharper transition)
    mid_tone_boost : float
        Strength of mid-tone emphasis (0.0 = no boost, 1.0 = strong boost)
    mid_tone_sigma : float
        Width of the mid-tone Gaussian bump (smaller = narrower, larger = wider)
    
    Returns
    -------
    importance : np.ndarray
        Importance map in [0, 1]; higher = more stipples (dark areas and mid-tones get higher importance)
    """
    I = np.clip(gray_img, 0.0, 1.0)
    
    # Invert brightness: dark areas should get more dots (higher importance)
    I_inverted = 1.0 - I
    
    # Create smooth downweighting mask for extreme tones
    # Downweight very dark regions (I < extreme_threshold_low)
    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask = dark_mask / dark_mask.max()
    
    # Downweight very light regions (I > extreme_threshold_high)
    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask = light_mask / light_mask.max()
    
    # Combine both masks
    extreme_mask = np.maximum(dark_mask, light_mask)
    
    # Apply smooth downweighting
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
    
    # Add smooth gradual mid-tone boost (Gaussian centered on 0.65)
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    
    # Boost importance in mid-tone regions
    importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
    
    # Normalize to [0,1]
    m, M = importance.min(), importance.max()
    if M > m: 
        importance = (importance - m) / (M - m)
    return importance
```
:::


### R


::: {.cell}

```{.r .cell-code}
compute_importance <- function(gray_img, 
                                extreme_downweight = 0.5,
                                extreme_threshold_low = 0.4,
                                extreme_threshold_high = 0.8,
                                extreme_sigma = 0.1,
                                mid_tone_boost = 0.4,
                                mid_tone_sigma = 0.2) {
  # Clip image to [0, 1]
  I <- pmax(pmin(gray_img, 1.0), 0.0)
  
  # Invert brightness
  I_inverted <- 1.0 - I
  
  # Dark mask
  dark_mask <- exp(-((I - 0.0)^2) / (2.0 * (extreme_sigma^2)))
  dark_mask[I >= extreme_threshold_low] <- 0.0
  if(max(dark_mask) > 0) {
    dark_mask <- dark_mask / max(dark_mask)
  }
  
  # Light mask
  light_mask <- exp(-((I - 1.0)^2) / (2.0 * (extreme_sigma^2)))
  light_mask[I <= extreme_threshold_high] <- 0.0
  if(max(light_mask) > 0) {
    light_mask <- light_mask / max(light_mask)
  }
  
  # Combine masks
  extreme_mask <- pmax(dark_mask, light_mask)
  
  # Apply downweighting
  importance <- I_inverted * (1.0 - extreme_downweight * extreme_mask)
  
  # Mid-tone boost
  mid_tone_center <- 0.65
  mid_tone_gaussian <- exp(-((I - mid_tone_center)^2) / (2.0 * (mid_tone_sigma^2)))
  if(max(mid_tone_gaussian) > 0) {
    mid_tone_gaussian <- mid_tone_gaussian / max(mid_tone_gaussian)
  }
  
  importance <- importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
  
  # Normalize
  m <- min(importance)
  M <- max(importance)
  if(M > m) {
    importance <- (importance - m) / (M - m)
  }
  
  return(importance)
}
```
:::


:::

## Blue Noise Stippling Algorithm

The stippling algorithm uses a modified void-and-cluster approach that:

1. Creates an importance map that identifies visually important regions
2. Initializes an energy field based on the importance map (higher importance → lower energy)
3. Uses a toroidal (periodic) Gaussian kernel for repulsion to ensure blue noise properties
4. Iteratively selects points with minimum energy
5. Adds Gaussian "splats" around selected points to prevent clustering
6. Balances image content importance with blue noise spatial distribution

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np

# compute_importance function is defined in the importance-map chunk
# void_and_cluster will use it from there

def toroidal_gaussian_kernel(h: int, w: int, sigma: float):
    """
    Create a periodic (toroidal) 2D Gaussian kernel centered at (0,0).
    The toroidal property means the kernel wraps around at the edges,
    ensuring consistent repulsion behavior regardless of point location.
    """
    y = np.arange(h)
    x = np.arange(w)
    # Compute toroidal distances (minimum distance considering wrapping)
    dy = np.minimum(y, h - y)[:, None]
    dx = np.minimum(x, w - x)[None, :]
    # Compute Gaussian
    kern = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    s = kern.sum()
    if s > 0:
        kern /= s  # Normalize
    return kern


def void_and_cluster(
    input_img: np.ndarray,
    percentage: float = 0.08,
    sigma: float = 0.9,
    content_bias: float = 0.9,
    importance_img: np.ndarray | None = None,
    noise_scale_factor: float = 0.1,
):
    """
    Generate blue noise stippling pattern from input image using a modified
    void-and-cluster algorithm with content-weighted importance.
    """
    I = np.clip(input_img, 0.0, 1.0)
    h, w = I.shape

    # Compute or use provided importance map
    if importance_img is None:
        importance = compute_importance(I)
    else:
        importance = np.clip(importance_img, 0.0, 1.0)

    # Create toroidal Gaussian kernel for repulsion
    kernel = toroidal_gaussian_kernel(h, w, sigma)

    # Initialize energy field: lower energy → more likely to be picked
    energy_current = -importance * content_bias

    # Stipple buffer: start with white background; selected points become black dots
    final_stipple = np.ones_like(I)
    samples = []

    # Helper function to roll kernel to an arbitrary position
    def energy_splat(y, x):
        """Get energy contribution by rolling the kernel to position (y, x)."""
        return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)

    # Number of points to select
    num_points = int(I.size * percentage)

    # Choose first point near center with minimal energy
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
    energy_current[y0, x0] = np.inf  # Prevent reselection
    samples.append((y0, x0, I[y0, x0]))
    final_stipple[y0, x0] = 0.0  # Black dot

    # Iteratively place remaining points
    for i in range(1, num_points):
        # Add exploration noise that decreases over time
        exploration = 1.0 - (i / num_points) * 0.5  # Decrease from 1.0 to 0.5
        noise = np.random.normal(0.0, noise_scale_factor * content_bias * exploration, size=energy_current.shape)
        energy_with_noise = energy_current + noise

        # Find position with minimum energy (with noise for exploration)
        pos_flat = np.argmin(energy_with_noise)
        y = pos_flat // w
        x = pos_flat % w

        # Add Gaussian splat to prevent nearby points from being selected
        energy_current = energy_current + energy_splat(y, x)
        energy_current[y, x] = np.inf  # Prevent reselection

        # Record the sample
        samples.append((y, x, I[y, x]))
        final_stipple[y, x] = 0.0  # Black dot

    return final_stipple, np.array(samples)
```
:::


### R


::: {.cell}

```{.r .cell-code}
# Note: R implementation provided for completeness, but Python is strongly recommended
# The R implementation may have issues with the circular shift that need debugging
# If you encounter problems (e.g., tiling/repeating patterns), please use the Python version

toroidal_gaussian_kernel <- function(h, w, sigma) {
  y <- 0:(h-1)
  x <- 0:(w-1)
  
  # Compute toroidal distances
  dy <- pmin(y, h - y)
  dx <- pmin(x, w - x)
  
  # Create distance matrices
  dy_mat <- outer(dy, rep(1, w))
  dx_mat <- outer(rep(1, h), dx)
  
  # Compute Gaussian
  kern <- exp(-(dx_mat^2 + dy_mat^2) / (2.0 * sigma^2))
  kern <- kern / sum(kern)  # Normalize
  
  return(kern)
}

void_and_cluster <- function(input_img, 
                             percentage = 0.08,
                             sigma = 0.9,
                             content_bias = 0.9,
                             importance_img = NULL,
                             noise_scale_factor = 0.1) {
  # Clip image to [0, 1]
  I <- pmax(pmin(input_img, 1.0), 0.0)
  h <- nrow(I)
  w <- ncol(I)
  
  # Compute or use provided importance map
  if(is.null(importance_img)) {
    importance <- compute_importance(I)
  } else {
    importance <- pmax(pmin(importance_img, 1.0), 0.0)
  }
  
  # Create toroidal Gaussian kernel for repulsion
  kernel <- toroidal_gaussian_kernel(h, w, sigma)
  
  # Initialize energy field: lower energy → more likely to be picked
  energy_current <- -importance * content_bias
  
  # Stipple buffer: start with white background; selected points become black dots
  final_stipple <- matrix(1.0, nrow = h, ncol = w)
  samples <- list()
  
  # Helper function to roll kernel to an arbitrary position
  # This implements the equivalent of Python's np.roll(kernel, shift=y, axis=0) 
  # followed by np.roll(kernel, shift=x, axis=1)
  # Instead of rolling, we compute the toroidal distance directly for each position
  energy_splat <- function(y, x) {
    # Create a matrix of the same size as kernel
    # For each position (i, j), compute the toroidal distance from (y, x)
    # and use the kernel value at that distance
    
    # Create coordinate grids (1-indexed in R)
    i_grid <- matrix(rep(1:h, each = w), nrow = h, ncol = w)
    j_grid <- matrix(rep(1:w, times = h), nrow = h, ncol = w)
    
    # Convert to 0-indexed for distance calculation
    i_0 <- i_grid - 1
    j_0 <- j_grid - 1
    y_0 <- y - 1  # Convert y to 0-indexed
    x_0 <- x - 1  # Convert x to 0-indexed
    
    # Compute toroidal distances
    dy <- pmin(abs(i_0 - y_0), h - abs(i_0 - y_0))
    dx <- pmin(abs(j_0 - x_0), w - abs(j_0 - x_0))
    
    # The kernel is centered at (0,0) in 0-indexed coordinates
    # So we need to find which kernel position corresponds to distance (dy, dx)
    # The kernel position is (dy, dx) in 0-indexed
    # Convert back to 1-indexed for R indexing
    kern_i <- dy + 1
    kern_j <- dx + 1
    
    # Extract kernel values at these positions
    # Use matrix indexing: kernel[kern_i, kern_j] but need to handle vectorized indexing
    kern_rolled <- matrix(kernel[cbind(as.vector(kern_i), as.vector(kern_j))], 
                         nrow = h, ncol = w)
    
    return(kern_rolled)
  }
  
  # Number of points to select
  num_points <- as.integer(length(I) * percentage)
  
  # Choose first point near center with minimal energy
  cy <- as.integer(h / 2)
  cx <- as.integer(w / 2)
  r <- min(20, as.integer(h / 10), as.integer(w / 10))
  y_start <- max(1, cy - r)
  y_end <- min(h, cy + r)
  x_start <- max(1, cx - r)
  x_end <- min(w, cx + r)
  
  region <- energy_current[y_start:y_end, x_start:x_end]
  flat_idx <- which.min(region)
  y0 <- ((flat_idx - 1) %/% ncol(region)) + y_start
  x0 <- ((flat_idx - 1) %% ncol(region)) + x_start
  
  # Place first point
  energy_current <- energy_current + energy_splat(y0, x0)
  energy_current[y0, x0] <- Inf  # Prevent reselection
  samples[[1]] <- c(y0, x0, I[y0, x0])
  final_stipple[y0, x0] <- 0.0  # Black dot
  
  # Iteratively place remaining points
  for(i in 2:num_points) {
    # Add exploration noise that decreases over time
    exploration <- 1.0 - ((i - 1) / num_points) * 0.5  # Decrease from 1.0 to 0.5
    noise <- matrix(rnorm(h * w, 0, noise_scale_factor * content_bias * exploration), 
                    nrow = h, ncol = w)
    energy_with_noise <- energy_current + noise
    
    # Find position with minimum energy (with noise for exploration)
    pos_flat <- which.min(energy_with_noise)
    y <- ((pos_flat - 1) %/% w) + 1
    x <- ((pos_flat - 1) %% w) + 1
    
    # Add Gaussian splat to prevent nearby points from being selected
    energy_current <- energy_current + energy_splat(y, x)
    energy_current[y, x] <- Inf  # Prevent reselection
    
    # Record the sample
    samples[[i]] <- c(y, x, I[y, x])
    final_stipple[y, x] <- 0.0  # Black dot
  }
  
  # Convert samples list to matrix
  samples_matrix <- do.call(rbind, samples)
  
  return(list(stipple = final_stipple, samples = samples_matrix))
}
```
:::


:::

## Preparing the Working Image

Before generating the stippling pattern, we prepare the image by resizing if necessary and computing the importance map.

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np
from PIL import Image

# Load image (always reload to ensure chunk is self-contained)
# This works whether or not variables from previous chunk are available
img_path = 'florida_beach_picture.jpg'
original_img = Image.open(img_path)
if original_img.mode != 'L':
    original_img = original_img.convert('L')
img_array = np.array(original_img, dtype=np.float32) / 255.0

# Resize image if it's too large for faster processing
max_size = 512
if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
    scale = max_size / max(img_array.shape[0], img_array.shape[1])
    new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
    img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
    if img_resized_pil.mode != 'L':
        img_resized_pil = img_resized_pil.convert('L')
    img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
    print(f"Resized image from {img_array.shape} to {img_resized.shape} for processing")
else:
    img_resized = img_array.copy()

# Ensure img_resized is 2D grayscale
if len(img_resized.shape) > 2:
    img_resized = img_resized[:, :, 0]
elif len(img_resized.shape) == 2:
    pass
else:
    raise ValueError(f"Unexpected image shape: {img_resized.shape}")

print(f"Final image shape: {img_resized.shape} (should be 2D for grayscale)")

# Ensure compute_importance function is available
# It should be from importance-map chunk, but we check to be safe
try:
    compute_importance
except NameError:
    # Redefine if not available (shouldn't happen, but safety check)
    def compute_importance(
        gray_img: np.ndarray,
        extreme_downweight: float = 0.5,
        extreme_threshold_low: float = 0.4,
        extreme_threshold_high: float = 0.8,
        extreme_sigma: float = 0.1,
        mid_tone_boost: float = 0.4,
        mid_tone_sigma: float = 0.2,
    ):
        I = np.clip(gray_img, 0.0, 1.0)
        I_inverted = 1.0 - I
        dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
        dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
        if dark_mask.max() > 0:
            dark_mask = dark_mask / dark_mask.max()
        light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
        light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
        if light_mask.max() > 0:
            light_mask = light_mask / light_mask.max()
        extreme_mask = np.maximum(dark_mask, light_mask)
        importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
        mid_tone_center = 0.65
        mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
        if mid_tone_gaussian.max() > 0:
            mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
        importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
        m, M = importance.min(), importance.max()
        if M > m: 
            importance = (importance - m) / (M - m)
        return importance

# Compute importance map using default parameters
importance_map = compute_importance(
    img_resized,
    extreme_downweight=0.5,
    extreme_threshold_low=0.2,
    extreme_threshold_high=0.8,
    extreme_sigma=0.1
)
print("Importance map computed")
```


::: {.cell-output .cell-output-stdout}

```
Resized image from (1060, 1600) to (339, 512) for processing
Final image shape: (339, 512) (should be 2D for grayscale)
Importance map computed
```


:::
:::


### R


::: {.cell}

```{.r .cell-code}
# Resize image if needed
max_size <- 512
img_dims <- dim(original_img)
if(img_dims[1] > max_size || img_dims[2] > max_size) {
  scale <- max_size / max(img_dims[1], img_dims[2])
  new_size <- c(round(img_dims[1] * scale), round(img_dims[2] * scale))
  img_resized <- resize(original_img, new_size[1], new_size[2])
  cat("Resized image to", new_size[1], "x", new_size[2], "for processing\n")
} else {
  img_resized <- original_img
}

# Convert to matrix and normalize
img_matrix <- as.matrix(img_resized)
img_matrix <- img_matrix / max(img_matrix)

cat("Final image shape:", dim(img_matrix)[1], "x", dim(img_matrix)[2], "\n")

# Compute importance map
importance_map <- compute_importance(img_matrix)
cat("Importance map computed\n")
```
:::


:::

## Generating the Stippled Image

Now let's apply the stippling algorithm to create the blue noise stippling pattern.

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np
from PIL import Image

# Reload and prepare image (since chunks don't share state)
img_path = 'florida_beach_picture.jpg'
original_img = Image.open(img_path)
if original_img.mode != 'L':
    original_img = original_img.convert('L')
img_array = np.array(original_img, dtype=np.float32) / 255.0

# Resize if needed
max_size = 512
if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
    scale = max_size / max(img_array.shape[0], img_array.shape[1])
    new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
    img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
    if img_resized_pil.mode != 'L':
        img_resized_pil = img_resized_pil.convert('L')
    img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
else:
    img_resized = img_array.copy()

if len(img_resized.shape) > 2:
    img_resized = img_resized[:, :, 0]

# Compute importance map
def compute_importance(
    gray_img: np.ndarray,
    extreme_downweight: float = 0.5,
    extreme_threshold_low: float = 0.4,
    extreme_threshold_high: float = 0.8,
    extreme_sigma: float = 0.1,
    mid_tone_boost: float = 0.4,
    mid_tone_sigma: float = 0.2,
):
    I = np.clip(gray_img, 0.0, 1.0)
    I_inverted = 1.0 - I
    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask = dark_mask / dark_mask.max()
    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask = light_mask / light_mask.max()
    extreme_mask = np.maximum(dark_mask, light_mask)
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
    m, M = importance.min(), importance.max()
    if M > m: 
        importance = (importance - m) / (M - m)
    return importance

importance_map = compute_importance(
    img_resized,
    extreme_downweight=0.5,
    extreme_threshold_low=0.2,
    extreme_threshold_high=0.8,
    extreme_sigma=0.1
)

# Redefine functions if not available
try:
    void_and_cluster
except NameError:
    # Need to redefine void_and_cluster and dependencies
    def toroidal_gaussian_kernel(h: int, w: int, sigma: float):
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
        I = np.clip(input_img, 0.0, 1.0)
        h, w = I.shape
        if importance_img is None:
            importance = compute_importance(I)
        else:
            importance = np.clip(importance_img, 0.0, 1.0)
        kernel = toroidal_gaussian_kernel(h, w, sigma)
        energy_current = -importance * content_bias
        final_stipple = np.ones_like(I)
        samples = []
        def energy_splat(y, x):
            return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)
        num_points = int(I.size * percentage)
        cy, cx = h // 2, w // 2
        r = min(20, h // 10, w // 10)
        ys = slice(max(0, cy - r), min(h, cy + r))
        xs = slice(max(0, cx - r), min(w, cx + r))
        region = energy_current[ys, xs]
        flat = np.argmin(region)
        y0 = flat // (region.shape[1]) + (cy - r)
        x0 = flat % (region.shape[1]) + (cx - r)
        energy_current = energy_current + energy_splat(y0, x0)
        energy_current[y0, x0] = np.inf
        samples.append((y0, x0, I[y0, x0]))
        final_stipple[y0, x0] = 0.0
        for i in range(1, num_points):
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

# Generate stippling pattern
print("Generating blue noise stippling pattern...")
stipple_pattern, samples = void_and_cluster(
    img_resized,
    percentage=0.08,
    sigma=0.9,
    content_bias=0.9,
    importance_img=importance_map,
    noise_scale_factor=0.1
)

print(f"Generated {len(samples)} stipple points")
print(f"Stipple pattern shape: {stipple_pattern.shape}")
```


::: {.cell-output .cell-output-stdout}

```
Generating blue noise stippling pattern...
Generated 13885 stipple points
Stipple pattern shape: (339, 512)
```


:::
:::


### R


::: {.cell}

```{.r .cell-code}
# Generate stippling pattern
cat("Generating blue noise stippling pattern...\n")
stipple_result <- void_and_cluster(
  img_matrix,
  percentage = 0.08,
  sigma = 0.9,
  content_bias = 0.9,
  importance_img = importance_map,
  noise_scale_factor = 0.1
)

stipple_pattern <- stipple_result$stipple
samples <- stipple_result$samples

cat("Generated", nrow(samples), "stipple points\n")
cat("Stipple pattern shape:", nrow(stipple_pattern), "x", ncol(stipple_pattern), "\n")
```
:::


:::

## Displaying the Results

Let's visualize the original image, the importance map, and the stippled version side by side for comparison.

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Reload and prepare image (since chunks don't share state)
img_path = 'florida_beach_picture.jpg'
original_img = Image.open(img_path)
if original_img.mode != 'L':
    original_img = original_img.convert('L')
img_array = np.array(original_img, dtype=np.float32) / 255.0

# Resize if needed
max_size = 512
if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
    scale = max_size / max(img_array.shape[0], img_array.shape[1])
    new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
    img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
    if img_resized_pil.mode != 'L':
        img_resized_pil = img_resized_pil.convert('L')
    img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
else:
    img_resized = img_array.copy()

if len(img_resized.shape) > 2:
    img_resized = img_resized[:, :, 0]

# Compute importance map
def compute_importance(gray_img, extreme_downweight=0.5, extreme_threshold_low=0.4, 
                       extreme_threshold_high=0.8, extreme_sigma=0.1, mid_tone_boost=0.4, mid_tone_sigma=0.2):
    I = np.clip(gray_img, 0.0, 1.0)
    I_inverted = 1.0 - I
    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask = dark_mask / dark_mask.max()
    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask = light_mask / light_mask.max()
    extreme_mask = np.maximum(dark_mask, light_mask)
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
    m, M = importance.min(), importance.max()
    if M > m: 
        importance = (importance - m) / (M - m)
    return importance

importance_map = compute_importance(img_resized, extreme_downweight=0.5, extreme_threshold_low=0.2, 
                                    extreme_threshold_high=0.8, extreme_sigma=0.1)

# Try to use stipple_pattern from generate-stipple chunk if available
# If not, we'll show a placeholder
try:
    stipple_pattern
    has_stipple = True
except NameError:
    has_stipple = False
    # Create a simple placeholder
    stipple_pattern = np.ones_like(img_resized)

if has_stipple:
    fig, axes = plt.subplots(1, 3, figsize=(7, 4))
else:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Display original image
axes[0].imshow(img_resized, cmap='gray', vmin=0, vmax=1)
axes[0].axis('off')
axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)

# Display importance map
axes[1].imshow(importance_map, cmap='gray', vmin=0, vmax=1)
axes[1].axis('off')
axes[1].set_title('Importance Map', fontsize=14, fontweight='bold', pad=10)

# Display stippled image (if available)
if has_stipple:
    axes[2].imshow(stipple_pattern, cmap='gray', vmin=0, vmax=1)
    axes[2].axis('off')
    axes[2].set_title('Blue Noise Stippling', fontsize=14, fontweight='bold', pad=10)
else:
    # Show message that stippling should be generated in previous chunk
    axes[1].text(0.5, 0.5, 'Stippling pattern\nwill be shown here\n(after generate-stipple chunk)', 
                 ha='center', va='center', fontsize=12, transform=axes[1].transAxes)

plt.tight_layout()
# plt.show() removed - Quarto will handle figure display
```
:::


### R


::: {.cell}

```{.r .cell-code}
par(mfrow = c(1, 3), mar = c(2, 2, 2, 2))

plot(img_resized, axes = FALSE, main = "Original Image")
plot(as.cimg(importance_map), axes = FALSE, main = "Importance Map")
plot(as.cimg(stipple_pattern), axes = FALSE, main = "Blue Noise Stippling")
```
:::


:::

## Progressive Stippling Animation

This section creates a GIF showing how the stippled image looks as more points are added sequentially. We'll use the already-computed stippling points to generate frames at increments of 100 points.

::: {.panel-tabset}

### Python


::: {.cell python.reticulate='false'}

```{.python .cell-code}
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# Reload and prepare image (since chunks don't share state)
img_path = 'florida_beach_picture.jpg'
original_img = Image.open(img_path)
if original_img.mode != 'L':
    original_img = original_img.convert('L')
img_array = np.array(original_img, dtype=np.float32) / 255.0

# Resize if needed
max_size = 512
if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
    scale = max_size / max(img_array.shape[0], img_array.shape[1])
    new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
    img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
    if img_resized_pil.mode != 'L':
        img_resized_pil = img_resized_pil.convert('L')
    img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
else:
    img_resized = img_array.copy()

if len(img_resized.shape) > 2:
    img_resized = img_resized[:, :, 0]

# Check if samples are available from generate-stipple chunk
# If not, we'll regenerate the stippling (since chunks don't share state)
try:
    samples
    has_samples = True
    print("Using samples from generate-stipple chunk")
except NameError:
    has_samples = False
    print("Samples not found. Regenerating stippling for progressive animation...")
    
    # Redefine functions needed for stippling
    def compute_importance(gray_img, extreme_downweight=0.5, extreme_threshold_low=0.4, 
                           extreme_threshold_high=0.8, extreme_sigma=0.1, mid_tone_boost=0.4, mid_tone_sigma=0.2):
        I = np.clip(gray_img, 0.0, 1.0)
        I_inverted = 1.0 - I
        dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
        dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
        if dark_mask.max() > 0:
            dark_mask = dark_mask / dark_mask.max()
        light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
        light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
        if light_mask.max() > 0:
            light_mask = light_mask / light_mask.max()
        extreme_mask = np.maximum(dark_mask, light_mask)
        importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
        mid_tone_center = 0.65
        mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
        if mid_tone_gaussian.max() > 0:
            mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
        importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
        m, M = importance.min(), importance.max()
        if M > m: 
            importance = (importance - m) / (M - m)
        return importance
    
    def toroidal_gaussian_kernel(h, w, sigma):
        y = np.arange(h)
        x = np.arange(w)
        dy = np.minimum(y, h - y)[:, None]
        dx = np.minimum(x, w - x)[None, :]
        kern = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
        s = kern.sum()
        if s > 0:
            kern /= s
        return kern
    
    def void_and_cluster(input_img, percentage=0.08, sigma=0.9, content_bias=0.9, 
                         importance_img=None, noise_scale_factor=0.1):
        I = np.clip(input_img, 0.0, 1.0)
        h, w = I.shape
        if importance_img is None:
            importance = compute_importance(I)
        else:
            importance = np.clip(importance_img, 0.0, 1.0)
        kernel = toroidal_gaussian_kernel(h, w, sigma)
        energy_current = -importance * content_bias
        final_stipple = np.ones_like(I)
        samples_list = []
        def energy_splat(y, x):
            return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)
        num_points = int(I.size * percentage)
        cy, cx = h // 2, w // 2
        r = min(20, h // 10, w // 10)
        ys = slice(max(0, cy - r), min(h, cy + r))
        xs = slice(max(0, cx - r), min(w, cx + r))
        region = energy_current[ys, xs]
        flat = np.argmin(region)
        y0 = flat // (region.shape[1]) + (cy - r)
        x0 = flat % (region.shape[1]) + (cx - r)
        energy_current = energy_current + energy_splat(y0, x0)
        energy_current[y0, x0] = np.inf
        samples_list.append((y0, x0, I[y0, x0]))
        final_stipple[y0, x0] = 0.0
        for i in range(1, num_points):
            exploration = 1.0 - (i / num_points) * 0.5
            noise = np.random.normal(0.0, noise_scale_factor * content_bias * exploration, size=energy_current.shape)
            energy_with_noise = energy_current + noise
            pos_flat = np.argmin(energy_with_noise)
            y = pos_flat // w
            x = pos_flat % w
            energy_current = energy_current + energy_splat(y, x)
            energy_current[y, x] = np.inf
            samples_list.append((y, x, I[y, x]))
            final_stipple[y, x] = 0.0
        return final_stipple, np.array(samples_list)
    
    # Compute importance map and generate stippling
    importance_map = compute_importance(img_resized, extreme_downweight=0.5, 
                                        extreme_threshold_low=0.2, extreme_threshold_high=0.8, extreme_sigma=0.1)
    print("Generating stippling pattern for progressive animation...")
    stipple_pattern, samples = void_and_cluster(img_resized, percentage=0.08, sigma=0.9, 
                                                content_bias=0.9, importance_img=importance_map, noise_scale_factor=0.1)
    has_samples = True
    print(f"Generated {len(samples)} stipple points for progressive animation")

# Use the existing samples array from the already-computed stippling
if has_samples:
    print(f"Using existing stippling with {len(samples)} points")
else:
    print("Note: Progressive stippling requires samples from generate-stipple chunk.")
    print("This chunk will create frames, but samples should be generated first.")
print(f"Image shape: {img_resized.shape}")

# Create progressive frames by adding points sequentially
frame_increment = 100
frames = []
point_counts = []

# Start with white background
h, w = img_resized.shape
progressive_stipple = np.ones_like(img_resized)

# Add first point and save initial frame
if has_samples and len(samples) > 0:
    y0, x0, intensity0 = int(samples[0, 0]), int(samples[0, 1]), samples[0, 2]
    progressive_stipple[y0, x0] = 0.0
    frames.append(progressive_stipple.copy())
    point_counts.append(1)

# Add remaining points sequentially and save frames at increments
if has_samples:
    for i in range(1, len(samples)):
        y, x = int(samples[i, 0]), int(samples[i, 1])
        progressive_stipple[y, x] = 0.0  # Add black dot
        
        # Save frame at increments (100, 200, 300, ...) and at the end
        if (i + 1) % frame_increment == 0 or i == len(samples) - 1:
            frames.append(progressive_stipple.copy())
            point_counts.append(i + 1)

print(f"Generated {len(frames)} frames")
print(f"Point counts: {point_counts}")
```


::: {.cell-output .cell-output-stdout}

```
Samples not found. Regenerating stippling for progressive animation...
Generating stippling pattern for progressive animation...
Generated 13885 stipple points for progressive animation
Using existing stippling with 13885 points
Image shape: (339, 512)
Generated 140 frames
Point counts: [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13885]
```


:::
:::


### R


::: {.cell}

```{.r .cell-code}
# Use the existing samples from the already-computed stippling
cat("Using existing stippling with", nrow(samples), "points\n")
cat("Image shape:", nrow(img_matrix), "x", ncol(img_matrix), "\n")

# Create progressive frames by adding points sequentially
frame_increment <- 100
frames <- list()
point_counts <- c()

# Start with white background
progressive_stipple <- matrix(1.0, nrow = nrow(img_matrix), ncol = ncol(img_matrix))

# Add first point and save initial frame
if(nrow(samples) > 0) {
  y0 <- as.integer(samples[1, 1])
  x0 <- as.integer(samples[1, 2])
  progressive_stipple[y0, x0] <- 0.0
  frames[[1]] <- progressive_stipple
  point_counts <- c(point_counts, 1)
}

# Add remaining points sequentially and save frames at increments
for(i in 2:nrow(samples)) {
  y <- as.integer(samples[i, 1])
  x <- as.integer(samples[i, 2])
  progressive_stipple[y, x] <- 0.0  # Add black dot
  
  # Save frame at increments (100, 200, 300, ...) and at the end
  if(i %% frame_increment == 0 || i == nrow(samples)) {
    frames[[length(frames) + 1]] <- progressive_stipple
    point_counts <- c(point_counts, i)
  }
}

cat("Generated", length(frames), "frames\n")
cat("Point counts:", paste(point_counts, collapse = ", "), "\n")
```
:::


:::

Now let's create the GIF animation:

::: {.panel-tabset}

### Python



### R



:::

![Progressive stippling animation showing the sequential build-up of points. Each frame represents an increment of 100 points, demonstrating how the blue noise stippling pattern develops as more points are added.](progressive_stippling.gif)

## Challenge Requirements 📋

### Your Task

**Part 1 (This Challenge):** Reproduce the blue noise stippling process to create:

1. **A stippled version of your chosen image** - Use the algorithm demonstrated above to create a stippled version of an image of your choice
2. **A progressive stippling GIF animation** - Create an animated GIF showing how the stippling pattern builds up as points are added
3. **GitHub Pages site** - Post both the stippled image and GIF to a GitHub Pages site with:
   - Appropriate captions explaining what the images show
   - A brief explanation of blue noise stippling and why it creates aesthetically pleasing patterns

**Part 2 (November 18th):** We'll use your stippled images to create a statistical meme about selection bias and missing data. More details will be provided in Part 2.

### Minimum Requirements for Any Points

1. **Fork and Clone Repository:** Use the starter repository (if provided) or create your own GitHub repository for this challenge

2. **Reproduce the Stippling Process:** 
   - Implement the blue noise stippling algorithm (Python recommended, R acceptable)
   - Create a stippled version of your chosen image
   - Generate a progressive stippling GIF animation

3. **GitHub Pages Setup:**
   - Create a GitHub Pages site showcasing your work
   - Include the stippled image with appropriate caption
   - Include the progressive GIF with appropriate caption
   - Add a brief explanation of blue noise stippling

4. **Code Organization:**
   - Well-organized, readable code
   - Comments explaining key steps
   - Use just one language (Python or R) - delete the other language sections

### Grading Rubric 🎓

**75% Grade Requirements:**
- Successfully reproduce the stippling algorithm
- Create a stippled image and progressive GIF
- Basic GitHub Pages site with images and captions

**85% Grade Requirements:**
- All of the above, plus:
- Clear explanation of how blue noise stippling works
- Discussion of why the algorithm creates aesthetically pleasing patterns
- Professional presentation of results

**95% Grade Requirements:**
- All of the above, plus:
- Experimentation with different parameters (percentage, sigma, etc.)
- Discussion of how parameter choices affect the final result
- Comparison of different parameter settings

**100% Grade Requirements:**
- All of the above, plus:
- Exceptional presentation and documentation
- Creative use of the stippling technique
- Insightful analysis of the algorithm's behavior
- Professional-quality GitHub Pages site

## Getting Started Tips 🚀

::: {.callout-note}
## 🎯 Navy SEALs Motto

> "Slow is Smooth and Smooth is Fast"

*Take your time to understand the stippling algorithm, plan your approach carefully, and execute with precision. Rushing through this challenge will only lead to errors and confusion.*
:::

::: {.callout-warning}
## 💾 Important: Save Your Work Frequently!

**Before you start:** Make sure to commit your work often using the Source Control panel in Cursor (Ctrl+Shift+G or Cmd+Shift+G). This prevents the AI from overwriting your progress and ensures you don't lose your work.

**Commit after each major step:**
- After implementing the importance map function
- After implementing the stippling algorithm
- After creating your stippled image
- After creating your GIF animation
- After setting up GitHub Pages

*Remember: Frequent commits are your safety net!*
:::

## Submission Checklist ✅

**Minimum Requirements (Required for Any Points):**

- [ ] Forked/cloned repository for this challenge
- [ ] Implemented blue noise stippling algorithm
- [ ] Created stippled version of your image
- [ ] Generated progressive stippling GIF animation
- [ ] Created GitHub Pages site
- [ ] Posted stippled image with caption
- [ ] Posted progressive GIF with caption
- [ ] Added brief explanation of blue noise stippling

**75% Grade Requirements:**

- [ ] All minimum requirements met
- [ ] Code is well-organized and readable
- [ ] GitHub Pages site is functional

**85% Grade Requirements:**

- [ ] All 75% requirements met
- [ ] Clear explanation of how blue noise stippling works
- [ ] Discussion of why the algorithm creates aesthetically pleasing patterns

**95% Grade Requirements:**

- [ ] All 85% requirements met
- [ ] Experimentation with different parameters
- [ ] Discussion of parameter effects

**100% Grade Requirements:**

- [ ] All 95% requirements met
- [ ] Exceptional presentation and documentation
- [ ] Creative use of the stippling technique
- [ ] Professional-quality GitHub Pages site


