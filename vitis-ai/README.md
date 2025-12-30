# Vitis-AI FPGA Deployment

This directory contains files and utilities for deploying the quantized U-Net model on Xilinx FPGA platforms using Vitis-AI.

## Contents

### Unet5_int.xmodel
- **Description:** INT8 quantized model compiled for Xilinx Deep Learning Processor (DPU)
- **Quantization:** Post-training quantization from FP32 to INT8
- **Target Architecture:** `DPUCZDX8G`

### inspector.py
Model inspection and validation utilities for Vitis-AI deployment. This was used during development to make sure the inference can fully run on the DPU without using the CPU.

## Vitis-AI Environment
```bash
# Clone Vitis-AI repository
git clone https://github.com/Xilinx/Vitis-AI
cd Vitis-AI

# Launch Docker container
./docker_run.sh xilinx/vitis-ai-cpu:latest
```

## References
- [DPU Product Guide](https://docs.xilinx.com/r/en-US/pg338-dpu)
- Xilinx Model Zoo for reference implementations


