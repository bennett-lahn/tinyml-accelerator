# TinyML Accelerator

A machine learning accelerator for embedded applications written in SystemVerilog, featuring a compact 4×4 output-stationary systolic array optimized for resource-constrained edge AI applications.

## Architecture Overview

The TinyML accelerator implements a systolic array-based CNN accelerator designed around a custom CIFAR-10 model. The design prioritizes memory efficiency and data reuse over raw computational throughput, making it suitable for embedded systems with strict resource constraints.

### Core Design Features

- **4×4 Output-Stationary Systolic Array**: 16 processing elements with 4-wide MAC units, capable of processing 4 input channels simultaneously
- **High-Efficiency Data Reuse**: Achieves 5.22× average activation reuse through intelligent memory organization, as well as 16× weight reuse
- **Compact Memory Footprint**: Total 46.25 KiB memory requirement (much smaller than typical embedded systems)
- **Quantized Arithmetic**: 8-bit quantization for weights and activations, 32-bit for biases and accumulators
- **Integrated Processing Pipeline**: On-the-fly pooling, requantization, and ReLU6 activation
- **Specialized Acceleration Units**: Dedicated units for fully connected layers and softmax computation

### Target Model Architecture

The accelerator is optimized for a custom CNN trained on grayscale CIFAR-10 data:

| Layer | Type | Input → Output | Output Size | Filter |
|-------|------|---------------|-------------|---------|
| 1 | Conv + Pool | 1→8 channels | 32×32 → 16×16 | 4×4, stride 1 |
| 2 | Conv + Pool | 8→16 channels | 16×16 → 8×8 | 4×4, stride 1 |
| 3 | Conv + Pool | 16→32 channels | 8×8 → 4×4 | 4×4, stride 1 |
| 4 | Conv + Pool | 32→64 channels | 4×4 → 2×2 | 4×4, stride 1 |
| 5 | FC | 256→256 | 256 | - |
| 6 | FC + Softmax | 256→10 | 10 | - |

### Key Components

- **`tensor_ram.sv`**: Banked memory system with 128-bit word organization for input data and activations
  - Two 2 KiB banked tensor RAMs (128-bit read, 8-bit write capability)
  - Row-major, channel-last data organization for optimal spatial locality
- **`TPU_Datapath.sv`**: Main processing unit coordinating all components
- **Systolic Array**: 4×4 output-stationary array with 4-wide processing elements
  - Performs 16,384 MAC operations in 40 cycles (vs 160 MACs for ARM Cortex-M52)
  - Processes 32 8-bit inputs per cycle at peak throughput
- **Weight ROM**: 42.125 KiB ROM with optimized indexing for 4-wide input channel groups
- **Universal Buffer**: Manages patch extraction for 7×7×4 activation windows
- **Output Coordinator**: Handles result accumulation with integrated pooling
- **FC Unit**: Dedicated 1-wide MAC unit for fully connected layers
- **Softmax Unit**: Lookup table-based exponential with numerical stability features

### Performance Characteristics

- **Memory Efficiency**: 5.22× average activation reuse, minimizing off-chip memory accesses
- **Parallelism**: Exploits spatial parallelism across input channels with 4-wide processing elements
- **Throughput**: 16,384 MACs in 40 cycles per output tile
- **Memory Footprint**: 46.25 KiB total (compatible with typical embedded systems)
- **Target Accuracy**: >90% on quantized CIFAR-10 classification (software validation)

## Dependencies

To run this project, you need:

- **Python 3.7+**
- **cocotb**: Hardware verification framework
  ```bash
  pip install cocotb
  ```
- **Icarus Verilog** or **Verilator**: SystemVerilog simulator (we only tested with Verilator and highly recommend using Verilator alongside the included Makefile)
  ```bash
  # For Ubuntu/Debian
  sudo apt-get install iverilog
  # Or for Verilator
  sudo apt-get install verilator
  ```
- **Make**: Build system
- **SystemVerilog-compatible simulator**

## Running the Project

### Basic Test Execution

The main test suite can be run using the TPU Datapath test:

```bash
# Navigate to the testbenches directory
cd testbenches

# Run the main test (adjust simulator as needed)
make SIM=icarus MODULE=test_TPU_Datapath TOPLEVEL=TPU_Datapath
```

### Test Architecture

The `test_TPU_Datapath.py` test simulates a complete inference pipeline through the custom CIFAR-10 model:

1. **Conv Layer 1**: 32×32×1 → 16×16×8 (4×4 filter, same padding, 2×2 max pool)
2. **Conv Layer 2**: 16×16×8 → 8×8×16 (4×4 filter, same padding, 2×2 max pool)  
3. **Conv Layer 3**: 8×8×16 → 4×4×32 (4×4 filter, same padding, 2×2 max pool)
4. **Conv Layer 4**: 4×4×32 → 2×2×64 (4×4 filter, same padding, 2×2 max pool)
5. **Flattening**: 2×2×64 → 256 vector
6. **Dense Layer 1**: 256 → 256 (fully connected with ReLU6)
7. **Dense Layer 2**: 256 → 10 (fully connected for classification)
8. **Softmax**: 10 logits → probability distribution

### Test Execution Flow

The test orchestrates complex control sequences for each layer:
- **Convolutional Layers**: Iterates through output channels, spatial tiles, and input channel groups
- **Systolic Array Coordination**: Manages patch extraction, weight loading, and MAC computation cycles
- **Memory Management**: Alternates between tensor RAMs for input/output to prevent blocking
- **Integration Testing**: Validates end-to-end inference including requantization and activation functions

### Customizing Input Data

To change the initial input data for testing:

#### 1. Update Tensor RAM Initialization

The tensor RAM can be initialized from a hex file. To modify the input:

1. **Create/Edit Hex File**: Create a hex file with your input data
   ```
   # Example: input_data.hex
   # Each line represents a 128-bit word in hexadecimal
   00010203040506070809101112131415
   16171819202122232425262728293031
   ...
   ```

2. **Update Module Instantiation**: Modify the tensor RAM instantiation to use your hex file:
   ```systemverilog
   tensor_ram #(
       .DEPTH_128B_WORDS(128),
       .INIT_FILE("path/to/your/input_data.hex")  // Add this parameter
   ) your_tensor_ram_instance (
       // ... port connections
   );
   ```

#### 2. Hex File Format

- Each line represents one 128-bit word
- Format: 32 hexadecimal characters per line (no spaces or 0x prefix)
- Byte ordering: MSB first (leftmost bytes are highest addresses)
- File should contain `DEPTH_128B_WORDS` lines

#### 3. Data Layout

The tensor RAM stores data in row-major, channel-last order optimized for the 4-wide processing elements:
```
Address = (row × num_cols × num_channels) + (col × num_channels) + channel
```

**For 32×32×1 input (Layer 1):**
- Pixel (0,0) channel 0 → Address 0
- Pixel (0,1) channel 0 → Address 1  
- Pixel (1,0) channel 0 → Address 32

**For multi-channel layers (e.g., 16×16×8):**
- Pixel (0,0) channels 0-3 → Addresses 0-3 (loaded in single 128-bit read)
- Pixel (0,0) channels 4-7 → Addresses 4-7
- Pixel (0,1) channels 0-3 → Addresses 8-11

This organization enables efficient 16-byte reads that align with the 4-wide processing elements, maximizing memory bandwidth utilization.

### Generating Custom Input Data

The `fakemodel/` directory contains tools for generating test inputs and model data:

#### Converting Images to Test Vectors

Use `convert.py` to convert any image into the 32×32 grayscale hex format expected by the accelerator:

```bash
cd fakemodel
python convert.py input_image.jpg test_vector_custom.hex
```

This script:
- Resizes the image to 32×32 pixels using high-quality LANCZOS resampling
- Converts to grayscale (single channel)
- Outputs hex values in row-major order with 16 pixels per line
- Includes headers describing the conversion parameters

#### Model Training and Weight Generation

To retrain the model or generate updated weight files:

```bash
cd fakemodel
# Train model and generate quantized TFLite file
python TPU_target_model.py

# Generate convolution weight ROM .hex files
python generate_conv_weights_hex.py
```

The tools automatically handle:
- **Model Training**: Custom CIFAR-10 CNN with grayscale conversion
- **INT8 Quantization**: Full model quantization with calibration dataset
- **Weight Extraction**: TFLite model analysis and weight extraction
- **Hardware Formatting**: Conversion to ROM-compatible hex files with proper addressing

## Project Structure

```
tinyml-accelerator/
├── rtl/                        # SystemVerilog source files
│   ├── tensor_ram.sv           # Main memory system
│   ├── TPU_Datapath.sv         # Top-level datapath
│   └── sys_types.svh           # System type definitions
├── testbenches/                # Test files
│   └── test_TPU_Datapath.py    # Main test suite
├── fakemodel/                  # Model training, quantization, and data conversion
│   ├── TPU_target_model.py     # TensorFlow model definition and training
│   ├── convert.py              # Image to hex converter for test inputs
│   ├── simple_cnn_32x32_quant_int8.tflite # Quantized model file
│   ├── test_vector_*.hex       # Generated test input images (dog, plane, car)
└── README.md                   # This file
```

## Current Limitations and Known Issues

Based on our development experience, here are several important considerations:

### Accuracy Challenges
- **Quantization Impact**: Inaccuracies in 8-bit quantization can cause significant accuracy loss (up to 60% in some cases)
- **ROM Data Integrity**: Any errors in weight/bias loading or indexing can severely impact results
- **Cumulative Error**: Requantization across 6 layers can compound small errors
- **Dataset Conversion**: RGB→Grayscale conversion reduces achievable accuracy (99%→94% typical)

### Hardware Limitations  
- **Model Specificity**: Currently optimized only for the custom CIFAR-10 model architecture
- **Limited Programmability**: No instruction set architecture for arbitrary model support
- **Memory Constraints**: Fixed memory allocation may not suit larger models

## Testing Different Models

**Note**: The current design is highly specialized for the target CIFAR-10 model. Adapting to different models requires:

1. **Architecture Compatibility**: Ensure layer dimensions are compatible with 4×4 systolic array
2. **Memory Sizing**: Verify that weight ROM and activation memory can accommodate the new model
3. **Quantization Strategy**: Re-evaluate 8-bit quantization impact for the target model
4. **Control Logic**: Modify test sequences and control state machines for different layer configurations

This is not a straightforward process but we hope to make it easier in the future.

## Future Improvements

- **Improve Adaptability**: Make it easier to configure the accelerator for a different model
- **Mixed Precision**: Explore low-precision floating-point to reduce quantization loss
- **Instruction Set**: Develop RISC-like ISA for programmable model support  
- **Integration**: Host processor integration for offloading specific operations
- **Memory Optimization**: Implement activation caching for better data reuse

## Debug and Monitoring

The test includes comprehensive logging and monitoring:
- **Layer Progress**: Real-time tracking of convolution layer completion
- **Systolic Array Status**: Monitoring of patch validity, MAC completion, and data flow
- **Memory Access**: Validation of address bounds and data integrity
- **Final Results**: Q1.31 fixed-point to float conversion for probability analysis
- **Performance Metrics**: Cycle counting and throughput measurement

Use cocotb's logging levels to adjust debug verbosity. The test outputs final classification probabilities in both fixed-point and floating-point formats for analysis.