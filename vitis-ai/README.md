# Vitis-AI FPGA Deployment

This directory contains files and utilities for deploying the quantized U-Net model on Xilinx FPGA platforms using Vitis-AI.

## Contents

### Unet5_int.xmodel
- **Description:** INT8 quantized model compiled for Xilinx Deep Learning Processor (DPU)
- **Size:** 28.9 MB
- **Format:** Xilinx Model (.xmodel) for DPU execution
- **Quantization:** Post-training quantization from FP32 to INT8
- **Target:** Xilinx DPU-enabled platforms (e.g., ZCU102, ZCU104, Kria KV260)

### inspector.py
Model inspection and validation utilities for Vitis-AI deployment.

## Quantization Process

The model was quantized using Vitis-AI tools:

1. **Float Model Preparation**
   - Export PyTorch model to ONNX format
   - Verify ONNX model correctness

2. **Quantization**
   - Post-training quantization (PTQ)
   - Calibration using representative dataset
   - INT8 weight and activation quantization

3. **Compilation**
   - Target DPU architecture specification
   - Operator fusion and optimization
   - Generation of `.xmodel` file

## Expected Performance

### FPGA Acceleration Benefits
- **Inference Time:** Expected <10ms per frame (vs 167ms on CPU)
- **Throughput:** >100 FPS (vs 6 FPS on CPU)
- **Power Efficiency:** ~5-10W total system power
- **Latency:** Low and deterministic for real-time applications

### Accuracy
INT8 quantization typically maintains:
- <1% accuracy degradation
- MSE expected ~0.57-0.58 (vs 0.567 FP32)

## Deployment Requirements

### Hardware
- Xilinx FPGA with DPU (e.g., ZCU102, ZCU104, Kria KV260)
- Minimum DPU configuration: B4096 or equivalent
- Sufficient FPGA resources for 7.2M parameter model

### Software
- Vitis-AI 3.0 or later
- PYNQ (if using PYNQ-compatible boards)
- DPU runtime libraries
- Python 3.8+ with NumPy

## Deployment Steps

### 1. Setup Vitis-AI Environment
```bash
# Clone Vitis-AI repository
git clone https://github.com/Xilinx/Vitis-AI
cd Vitis-AI

# Launch Docker container
./docker_run.sh xilinx/vitis-ai-cpu:latest
```

### 2. Prepare Target Platform
```bash
# Flash SD card with Vitis-AI board image
# Copy .xmodel file to target board
scp Unet5_int.xmodel root@<board-ip>:/home/root/
```

### 3. Run Inference on Target
```python
import numpy as np
from vitis_ai_library import xir

# Load the model
graph = xir.Graph.deserialize("Unet5_int.xmodel")

# Create DPU runner
dpu_runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)

# Prepare input (256, 1024) -> DPU format
input_data = preprocess(raw_signal)

# Run inference
output = dpu_runner.execute_async([input_data])

# Post-process output (128, 128)
reconstructed_image = postprocess(output)
```

## Model Inspection

Use `inspector.py` to validate the compiled model:

```bash
python inspector.py --model Unet5_int.xmodel
```

This will display:
- Model architecture
- Layer-by-layer specifications
- DPU operator mapping
- Memory requirements
- Expected performance metrics

## DPU Configuration

### Required DPU Specs
- **RAM:** >100 MB for model parameters
- **Architecture:** B4096 or higher recommended
- **Operating Frequency:** 300+ MHz
- **INT8 Support:** Required

### Supported Operations
The U-Net model uses DPU-compatible operations:
- Convolution (Conv2d)
- Batch Normalization (fused with Conv)
- ReLU activation
- Max Pooling
- Transposed Convolution (ConvTranspose2d)
- Element-wise addition (skip connections)

## Optimization Notes

### Architecture Modifications for DPU
The U-Net architecture was designed with DPU constraints:
- Skip connections use strided convolution (not cropping)
- Batch normalization fused with convolution layers
- No dynamic control flow
- Fixed input/output dimensions

### Potential Optimizations
- **Pruning:** Reduce model size by 40-60% with minimal accuracy loss
- **Architecture search:** Find optimal depth/width trade-offs
- **Mixed precision:** Keep critical layers in INT16

## Performance Profiling

To profile the model on target hardware:

```bash
# Run with profiling enabled
vaitrace --cmd "python inference.py"

# Analyze results
vaitrace_parse ./vaitrace_result.txt
```

This will show:
- Layer-wise execution time
- DPU utilization
- Memory bandwidth usage
- Bottleneck identification

## Troubleshooting

### Model Won't Compile
- Verify ONNX opset compatibility (use opset 11)
- Check for unsupported operations
- Ensure fixed input dimensions

### Low Accuracy After Quantization
- Use more calibration images (recommended: 100-1000)
- Try quantize-aware training (QAT) instead of PTQ
- Check for outliers in activation distributions

### Poor Performance
- Verify DPU is running at expected frequency
- Check for memory bandwidth bottlenecks
- Consider model pruning to reduce parameter count

## References

- [Vitis-AI Documentation](https://xilinx.github.io/Vitis-AI/)
- [DPU IP Product Guide](https://docs.xilinx.com/r/en-US/pg338-dpu)
- Xilinx Model Zoo for reference implementations

## Contact

For deployment assistance or performance optimization, refer to the project documentation or contact the development team.
