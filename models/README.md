# Model Implementations

This directory contains the neural network implementations for optoacoustic image reconstruction.

## Models

### unet5-256ch.ipynb
**Primary model for the project**

- **Architecture:** U-Net with 256-channel bottleneck
- **Parameters:** 7.2M trainable parameters
- **Input:** `(batch, 1, 256, 1024)` - Full 256-channel sensor array
- **Output:** `(batch, 1, 128, 128)` - Reconstructed image
- **Performance:**
  - Test MSE: 0.567
  - Inference: 167ms per frame (6 FPS on CPU)
  - Model size: 27.6 MB

**Architecture Details:**
```
Encoder:  1 → 32 → 64 → 128 → 256 (with max pooling)
Bottleneck: 256 → 512 → 256 (with spatial reduction)
Decoder: 256 → 128 → 64 → 32 → 1 (with skip connections)
```

**Key Features:**
- Double convolution blocks with batch normalization
- Skip connections with channel-wise convolution for spatial alignment
- Specialized bottleneck for 256×1024 → 128×128 transformation
- Compatible with Vitis-AI quantization

### unet6-128ch.ipynb
**Variant using only linear segment channels**

- **Architecture:** U-Net optimized for 128-channel input
- **Input:** `(batch, 1, 128, 1024)` - Linear segment only (channels 64-192)
- **Output:** `(batch, 1, 128, 128)` - Reconstructed image
- **Use case:** Comparison study, reduced computational complexity

## Training Procedure

Both models follow the same training pipeline:

1. **Data Loading:**
   ```python
   train_data_input, train_data_label, test_data_input, test_data_label = get_data()
   ```

2. **Training:**
   - Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
   - Loss: Mean Squared Error (MSE)
   - Batch size: 32
   - Epochs: 25
   - Device: CUDA (if available)

3. **Checkpointing:**
   - Best model saved based on lowest loss
   - Final model saved after each epoch

4. **Testing:**
   - Batch-wise inference to manage memory
   - MSE evaluation on test set
   - Visual output generation for qualitative assessment

## Model Files

Trained models are saved in the following formats:

- `best_unet5.pth` - PyTorch state dictionary (best validation loss)
- `unet5.pth` - PyTorch state dictionary (final epoch)
- `best_unet5-MSE-24ep.onnx` - ONNX format for deployment

## Usage

### Training
```python
# Initialize model
model = Unet5()

# Train
model = train_model(train_data_input, train_data_label)
```

### Inference
```python
# Load trained model
model = Unet5()
model.load_state_dict(torch.load('best_unet5.pth'))
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

### Export to ONNX
```python
dummy_input = torch.randn(1, 1, 256, 1024)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)
```

## Model Architecture Considerations

### Skip Connection Design
The skip connections use strided convolutions instead of cropping to match spatial dimensions:
```python
self.skip2 = nn.Conv2d(64, 64, kernel_size=(1,4), stride=(1,4))
self.skip3 = nn.Conv2d(128, 128, kernel_size=(1,4), stride=(1,4))
self.skip4 = nn.Conv2d(256, 256, kernel_size=(1,4), stride=(1,4))
```

This design choice ensures:
- Compatibility with Xilinx DPU constraints
- Efficient spatial alignment without cropping artifacts
- Preservation of semantic information across scales

### Bottleneck Optimization
The bottleneck layer performs both channel expansion and spatial reduction:
```python
Conv2d(256, 512, kernel_size=3)     # Expand channels
Conv2d(512, 512, kernel_size=(1,4), stride=(1,4))  # Reduce width
Conv2d(512, 256, kernel_size=3)     # Project back
```

This allows the network to:
- Compress 256×1024 input to 128×128 output
- Learn rich feature representations
- Maintain computational efficiency

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- torchinfo (for model summaries)
- tqdm (for progress bars)

## Notes

- Both notebooks include visualization utilities for debugging
- Training data is loaded from the `dataset5/` directory
- Models automatically use GPU if available
- Batch size can be adjusted based on available memory
