# Standard Reconstruction Methods

This directory contains the implementation of traditional optoacoustic image reconstruction algorithms for comparison with deep learning approaches.

## Contents

### standard-recon-2.ipynb
Implementation of delay-and-sum (back-projection) reconstruction algorithm.

## Reconstruction Method

### Delay-and-Sum Algorithm

The standard reconstruction uses the classic time-delay-and-sum approach:

1. **For each pixel in the reconstruction grid:**
   - Calculate distance to each transducer element
   - Compute time-of-flight delay: `delay = distance / speed_of_sound`
   - Sample the corresponding time point from each channel
   - Sum contributions from all channels

2. **Mathematical formulation:**
   ```
   I(x,y) = Σ s_i(t = d_i(x,y) / c)
   ```
   where:
   - `I(x,y)` is the reconstructed pixel intensity
   - `s_i(t)` is the time signal from channel i
   - `d_i(x,y)` is the distance from pixel (x,y) to sensor i
   - `c` is the speed of sound (1510 m/s)

## Reconstruction Parameters

### Sensor Array Configuration
- **Total channels:** 256
- **Array geometry:**
  - Left arc: 64 channels (38mm radius, 0.6mm pitch)
  - Linear segment: 128 channels (0.25mm pitch, 31.75mm total)
  - Right arc: 64 channels (38mm radius, 0.6mm pitch)
- **Sensor positions:** Loaded from `pos_sensors_multisegment.mat`

### Imaging Parameters
- **Sampling rate:** 40 MHz (25 ns per sample)
- **Time samples:** 1006 per channel
- **Speed of sound:** 1510 m/s
- **Max reconstruction distance:** ~38 mm

### Field of View
- **Size:** 31.75 × 31.75 mm (matches linear segment length)
- **Resolution:** 250 × 250 μm (from receiver bandwidth)
- **Grid dimensions:** 128 × 128 pixels
- **Position:** Centered on linear segment, extending below array

## Data Processing Pipeline

### 1. Load Raw Data
```python
file = h5py.File('sigMat_*.mat', 'r')
data = file.get('sigMat')  # Shape: (N_frames, 256, 1006)
```

### 2. Preprocessing
```python
# Remove DC offset
filtered = img - np.mean(img, axis=1, keepdims=True)

# Z-score normalization
normed = (filtered - np.mean(filtered)) / np.std(filtered)

# Convert to float32
normed = normed.astype(np.float32)
```

### 3. Reconstruction
```python
recon = reconstruction(piezo_pos, timeSig, sos, T, nX, nY, xVec, yVec, max_dis)
```

### 4. Post-processing
```python
# Normalize output
recon = (recon - np.mean(recon)) / np.std(recon)
```

## Dataset Generation

This notebook also includes the pipeline for generating training/testing datasets:

### Input Generation
1. Load raw `.mat` files from `large_scale_testing/data_mat/`
2. Apply preprocessing (DC removal, normalization, padding)
3. Split into train/test (90%/10%)
4. Save as `.npy` arrays: `(N, 1, 256, 1024)`

### Ground Truth Generation
1. For each preprocessed input image
2. Apply delay-and-sum reconstruction
3. Normalize the output
4. Save as `.npy` arrays: `(N, 1, 128, 128)`

### Data Sources
Four acquisition sessions:
- `sigMat_luis_leftarm_normal.mat`
- `sigMat_luis_leftarm_parallel.mat`
- `sigMat_luis_rightarm_normal.mat`
- `sigMat_luis_rightarm_parallel.mat`

Each file contains ~1400 frames after removing first 100 corrupted frames.

**Total dataset:**
- Training: 5040 samples
- Testing: 556 samples

## Computational Complexity

The delay-and-sum algorithm has complexity:
```
O(N_pixels × N_channels)
= O(128 × 128 × 256)
≈ 4.2M operations per frame
```

This is computationally expensive for real-time applications, motivating the deep learning approach.

## Comparison with Deep Learning

| Aspect | Delay-and-Sum | U-Net (DL) |
|--------|---------------|------------|
| Reconstruction time | ~seconds | 167 ms |
| Implementation | Explicit physics | Learned mapping |
| Image quality | Baseline | Similar (MSE: 0.567) |
| Real-time capable | No | Yes (6 FPS) |
| Hardware acceleration | Limited | FPGA/GPU friendly |

## Visualization

The notebook includes visualization utilities:
- Sensor array geometry plotting
- FOV grid visualization
- Input signal images (time-domain)
- Reconstructed images with proper spatial scaling

## Usage

Run the notebook cells sequentially:

1. Define sensor positions
2. Set reconstruction parameters
3. Load and preprocess data
4. Perform reconstruction
5. (Optional) Generate datasets for DL training

## Dependencies

- NumPy
- Matplotlib
- h5py (for `.mat` file loading)
- SciPy (for `.mat` files)

## References

The delay-and-sum algorithm is the standard method in photoacoustic imaging. For more details on the physics and theory, refer to the project presentation.
