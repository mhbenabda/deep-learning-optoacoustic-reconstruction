# Dataset Information

## Proprietary Notice

The optoacoustic imaging dataset used in this project is proprietary to the research laboratory and cannot be publicly shared. This directory serves as a placeholder for the dataset structure.

## Expected Dataset Structure

If you have access to the dataset, it should be organized as follows:

```
data/
├── dataset5/
│   ├── all_train/
│   │   ├── input/
│   │   │   └── train_input_all.npy      # (5040, 1, 256, 1024)
│   │   └── output/
│   │       └── train_label_all.npy      # (5040, 1, 128, 128)
│   └── all_test/
│       ├── input/
│       │   └── test_input_all.npy       # (556, 1, 256, 1024)
│       └── output/
│           └── test_label_all.npy       # (556, 1, 128, 128)
├── pos_sensors_multisegment.mat         # Sensor positions
└── large_scale_testing/
    └── data_mat/
        ├── sigMat_luis_leftarm_normal.mat
        ├── sigMat_luis_leftarm_parallel.mat
        ├── sigMat_luis_rightarm_normal.mat
        └── sigMat_luis_rightarm_parallel.mat
```

## Dataset Specifications

### Input Data
- **Format:** NumPy arrays (`.npy`)
- **Shape:** `(N_samples, 1, 256, 1024)`
  - N_samples: Number of images
  - 1: Single channel (grayscale)
  - 256: Number of transducer channels
  - 1024: Time samples (1006 signal + 18 median padding)
- **Preprocessing:**
  - DC offset removal (mean subtraction per channel)
  - Z-score normalization
  - Median padding (1006 → 1024 samples)
- **Data Type:** `float32`

### Output Data (Ground Truth)
- **Format:** NumPy arrays (`.npy`)
- **Shape:** `(N_samples, 1, 128, 128)`
  - 128×128: Reconstructed image dimensions
- **Generation:** Standard delay-and-sum reconstruction
- **Preprocessing:** Z-score normalization
- **Data Type:** `float32`

### Acquisition Parameters
- **Sampling Rate:** 40 MHz
- **Speed of Sound:** 1,510 m/s
- **Transducer Array:** 256 channels
  - 64 arc-left (38mm radius, 0.6mm pitch)
  - 128 linear segment (0.25mm pitch)
  - 64 arc-right (38mm radius, 0.6mm pitch)
- **Field of View:** 31.75×31.75 mm
- **Spatial Resolution:** 250×250 μm

## Dataset Generation

The dataset was generated from raw `.mat` files using the preprocessing pipeline in `standard_reconstruction/standard-recon-2.ipynb`:

1. Load raw signals from `.mat` files
2. Remove DC offset per channel
3. Apply z-score normalization
4. Generate ground truth using delay-and-sum reconstruction
5. Split into train/test (90%/10%)
6. Save as NumPy arrays

## Contact

For dataset access inquiries, please contact the laboratory at [contact information].
