# Results

This directory contains output visualizations and performance metrics from the trained U-Net models.

## Contents

### Sample Reconstructions
- `image_1.png` through `image_5.png` - Representative reconstruction examples
- `subchannel_reconstruction/` - Results from channel subset experiments

## Visualization Format

Each result image typically shows a comparison of:
1. **Input:** Raw time-domain sensor signals (256 channels × 1024 samples)
2. **Output:** U-Net reconstructed image (128 × 128 pixels)
3. **Ground Truth:** Standard delay-and-sum reconstruction (128 × 128 pixels)

## Performance Metrics

### Quantitative Results
- **Test MSE:** 0.567
- **Total Parameters:** 7.2M
- **Model Size:** 27.6 MB
- **Inference Time (CPU):** 167 ms per frame
- **Throughput:** 6 FPS

### Qualitative Assessment
The reconstructed images demonstrate:
- Preservation of fine anatomical details
- Accurate representation of spatial features
- Minimal artifacts compared to ground truth
- Consistent quality across test dataset

## Generating New Results

To generate results from a trained model:

```python
# Load model and test data
model = Unet5()
model.load_state_dict(torch.load('best_unet5.pth'))
test_data_input, test_data_label = load_test_data()

# Run inference
test_model(model, test_data_input, test_data_label)

# Results saved to dataset5/test_image_output/
```

The `test_model()` function in `models/unet5-256ch.ipynb` automatically:
- Performs batch-wise inference
- Calculates MSE metrics
- Generates comparison visualizations
- Saves outputs to disk

## Image Specifications

- **Format:** PNG
- **Colormap:** Grayscale
- **Spatial units:** mm (based on 250 μm resolution)
- **Value range:** Normalized (z-score)

## Subchannel Reconstruction

The `subchannel_reconstruction/` directory contains experiments with:
- Linear segment only (128 channels, indices 64-192)
- Arc segments only
- Various channel subset configurations

These experiments help understand:
- Contribution of different array segments
- Trade-offs between channels and reconstruction quality
- Potential for reduced sensor arrays

## Notes

- All visualizations use the same normalization for fair comparison
- Sample images are selected to represent diversity in the test set
- Full test set results are available upon request
