# Deep Learning for Optoacoustic Image Reconstruction

A deep learning approach for real-time optoacoustic (photoacoustic) image reconstruction using U-Net architecture, with FPGA deployment capability on Kria K260 for accelerated inference.

## Overview

This project implements a convolutional neural network (U-Net) to reconstruct optoacoustic images from raw time-domain sensor signals. The deep learning approach significantly reduces reconstruction time compared to traditional delay-and-sum methods while maintaining image quality.

**Key Features:**
- U-Net architecture with 7.2M parameters optimized for optoacoustic reconstruction
- Input: 256-channel × 1024-sample time-domain signals
- Output: 128×128 pixel reconstructed images
- Achieved MSE: 0.567 on test set
- ONNX export for deployment flexibility
- Vitis-AI quantized model for FPGA acceleration

## Project Structure

```
.
├── models/                      # Neural network implementations
│   ├── unet5-256ch.ipynb       # U-Net with 256-channel encoder
│   └── unet6-128ch.ipynb       # U-Net variant with 128 channels
├── standard_reconstruction/     # Traditional reconstruction methods
│   └── standard-recon-2.ipynb  # Time-delay-and-sum implementation
├── data/                        # Dataset directory (proprietary)
├── results/                     # Output images and metrics
├── vitis-ai/                    # FPGA deployment files
│   ├── Unet5_int.xmodel        # Quantized model for DPU
│   └── inspector.py            # Model inspection utilities
├── requirements.txt             # Python dependencies
└── presentation.pdf             # Project documentation
```

## Model Architecture

The U-Net architecture (`Unet5`) consists of:
- **Encoder:** 4 levels with double convolution blocks (32→64→128→256 channels)
- **Bottleneck:** 512-channel layer with spatial reduction
- **Decoder:** 4 levels with skip connections and transposed convolutions
- **Skip connections:** Spatial downsampling to match decoder dimensions
- **Normalization:** Batch normalization after each convolution
- **Activation:** ReLU throughout

**Input Shape:** `(batch, 1, 256, 1024)` - Raw sensor time signals
**Output Shape:** `(batch, 1, 128, 128)` - Reconstructed spatial images

## Dataset

The dataset consists of optoacoustic measurements from multi-segment transducer arrays:
- **Training:** 5,040 samples
- **Testing:** 556 samples
- **Preprocessing:** DC offset removal, z-score normalization, median padding
- **Ground Truth:** Generated using standard delay-and-sum reconstruction

Data acquisition details:
- 256-channel transducer array (64 arc-left + 128 linear + 64 arc-right)
- Sampling rate: 40 MHz
- Speed of sound: 1,510 m/s
- Field of view: 31.75×31.75 mm

**Note:** The dataset is proprietary and not included in this repository.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-learning-optoacoustic-reconstruction

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch with CUDA support (recommended)
- NumPy, Matplotlib, h5py
- torchinfo for model summaries
- Vitis-AI docker


## Results

| Metric | Value |
|--------|-------|
| Test MSE | 0.567 |
| Parameters | 7.2M |
| Model Size | 27.6 MB |
| Inference Time (CPU) | 167 ms |
| Throughput | 6 FPS |

Sample reconstructions are available in the `results/` directory.

## FPGA Deployment

The `vitis-ai/` directory contains files for deploying the quantized model on Xilinx FPGA platforms:
- `Unet5_int.xmodel`: INT8 quantized model for DPU execution
- `inspector.py`: Model inspection and validation tools

Refer to Xilinx Vitis-AI documentation for deployment instructions.

## Comparison with Standard Reconstruction

The `standard_reconstruction/` directory implements traditional delay-and-sum (back-projection) reconstruction for comparison. The DL approach achieves:
- Similar image quality (MSE: 0.567)
- Significantly faster inference
- Better suitability for real-time applications

## References

For more details, see `presentation.pdf` or the project description in my [portfolio](https://mhbenabda.github.io/projects/4_project/).

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Context

Developed as part of the SoCDAML project.
