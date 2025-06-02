#!/usr/bin/env python3
"""
Generate requantize scale ROM hex file for TinyML accelerator.

This script takes a TensorFlow Lite (.tflite) model as input and extracts
quantization parameters from Conv2D layers to generate the fixed-point
multiplier and shift format expected by the requantize_activate_unit.

The ROM layout is:
- NUM_LAYERS entries (6) for layer output scales
- Each entry: [31:0] multiplier, [37:32] shift (38-bit total)
"""

import numpy as np
import math
import struct
import argparse
import sys
import os

try:
    import tflite
except ImportError:
    print("Error: tflite package not found. Please install it with:")
    print("pip install tflite")
    sys.exit(1)

def quantization_params_to_multiplier_shift(scale, zero_point=0):
    """
    Convert TensorFlow Lite quantization scale to fixed-point multiplier and shift.
    
    This implements the algorithm used in TensorFlow Lite's quantization:
    - The scale is represented as M * 2^(-shift) where M is a 31-bit signed integer
    - M is in the range [2^30, 2^31-1] to maximize precision
    
    Args:
        scale: Float scale factor from TFLite quantization
        zero_point: Zero point (typically 0 for symmetric quantization)
    
    Returns:
        tuple: (multiplier, shift) where multiplier is int32 and shift is int (0-31)
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    
    # Find the shift such that scale * 2^shift is in range [0.5, 1.0)
    # This ensures the multiplier will be in range [2^30, 2^31-1]
    shift = 0
    scaled_value = scale
    
    # Scale up if too small
    while scaled_value < 0.5:
        scaled_value *= 2
        shift += 1
    
    # Scale down if too large  
    while scaled_value >= 1.0:
        scaled_value /= 2
        shift -= 1
    
    # Convert to 31-bit signed integer (Q31 format)
    # Multiply by 2^31 and round to nearest integer
    multiplier = int(round(scaled_value * (1 << 31)))
    
    # Ensure multiplier is in valid range [2^30, 2^31-1]
    if multiplier < (1 << 30):
        multiplier = 1 << 30
    elif multiplier >= (1 << 31):
        multiplier = (1 << 31) - 1
    
    # Convert to signed 32-bit integer
    if multiplier >= (1 << 31):
        multiplier = multiplier - (1 << 32)
    
    return multiplier, shift

def parse_tflite_model(model_path):
    """
    Parse a TensorFlow Lite model and extract Conv2D layer information.
    
    Args:
        model_path: Path to the .tflite model file
        
    Returns:
        list: List of dictionaries containing Conv2D layer information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the TFLite model
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    # Parse the model using tflite package
    model = tflite.Model.GetRootAsModel(model_data, 0)
    
    conv2d_layers = []
    
    # Get the first subgraph (most models have only one)
    if model.SubgraphsLength() == 0:
        raise ValueError("No subgraphs found in the model")
    
    subgraph = model.Subgraphs(0)
    
    # Iterate through operators to find Conv2D layers
    for op_idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(op_idx)
        
        # Get operator code
        op_code_idx = operator.OpcodeIndex()
        op_code = model.OperatorCodes(op_code_idx)
        
        # Check if this is a Conv2D operation
        builtin_code = op_code.BuiltinCode()
        if builtin_code == tflite.BuiltinOperator.CONV_2D:
            # Get input and output tensor indices
            input_idx = operator.Inputs(0)  # Input tensor
            output_idx = operator.Outputs(0)  # Output tensor
            
            # Get input tensor info
            input_tensor = subgraph.Tensors(input_idx)
            output_tensor = subgraph.Tensors(output_idx)
            
            # Extract quantization parameters
            input_quant = input_tensor.Quantization()
            output_quant = output_tensor.Quantization()
            
            # Get scales and zero points
            input_scale = None
            input_zero_point = None
            output_scale = None
            output_zero_point = None
            
            if input_quant and input_quant.ScaleLength() > 0:
                input_scale = input_quant.Scale(0)
                if input_quant.ZeroPointLength() > 0:
                    input_zero_point = input_quant.ZeroPoint(0)
                else:
                    input_zero_point = 0
            
            if output_quant and output_quant.ScaleLength() > 0:
                output_scale = output_quant.Scale(0)
                if output_quant.ZeroPointLength() > 0:
                    output_zero_point = output_quant.ZeroPoint(0)
                else:
                    output_zero_point = 0
            
            # Get tensor names
            input_name = input_tensor.Name().decode('utf-8') if input_tensor.Name() else f"input_{input_idx}"
            output_name = output_tensor.Name().decode('utf-8') if output_tensor.Name() else f"output_{output_idx}"
            
            # Get tensor shapes
            input_shape = []
            for i in range(input_tensor.ShapeLength()):
                input_shape.append(input_tensor.Shape(i))
            
            output_shape = []
            for i in range(output_tensor.ShapeLength()):
                output_shape.append(output_tensor.Shape(i))
            
            layer_info = {
                'layer_idx': op_idx,
                'input_name': input_name,
                'output_name': output_name,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'input_scale': input_scale,
                'input_zero_point': input_zero_point,
                'output_scale': output_scale,
                'output_zero_point': output_zero_point,
                'output_channels': output_shape[-1] if len(output_shape) >= 3 else 1
            }
            
            conv2d_layers.append(layer_info)
    
    return conv2d_layers

def generate_requantize_rom_hex(model_path, output_file="quant_params.hex"):
    """Generate the hex file for requantize scale ROM from a TFLite model."""
    
    # Parameters matching the SystemVerilog module
    NUM_LAYERS = 6
    MULT_WIDTH = 32
    SHIFT_WIDTH = 6
    
    print(f"Parsing TensorFlow Lite model: {model_path}")
    
    try:
        conv2d_layers = parse_tflite_model(model_path)
    except Exception as e:
        print(f"Error parsing TFLite model: {e}")
        return
    
    if not conv2d_layers:
        print("Warning: No Conv2D layers found in the model")
        conv2d_layers = []
    
    print(f"Found {len(conv2d_layers)} Conv2D layers")
    
    # Display layer information
    for i, layer in enumerate(conv2d_layers):
        print(f"\nLayer {i}:")
        print(f"  Input: {layer['input_name']} {layer['input_shape']}")
        print(f"  Output: {layer['output_name']} {layer['output_shape']}")
        print(f"  Input scale: {layer['input_scale']}, zero_point: {layer['input_zero_point']}")
        print(f"  Output scale: {layer['output_scale']}, zero_point: {layer['output_zero_point']}")
        print(f"  Output channels: {layer['output_channels']}")
    
    # Initialize ROM data
    rom_data = []
    
    # Add layer output scales
    print(f"\nAdding layer output scale entries...")
    
    # Process up to NUM_LAYERS Conv2D layers
    for layer_idx in range(NUM_LAYERS):
        if layer_idx < len(conv2d_layers):
            layer = conv2d_layers[layer_idx]
            
            # Use output scale for requantization
            if layer['output_scale'] is not None and layer['output_scale'] > 0:
                scale = layer['output_scale']
            else:
                print(f"Warning: Layer {layer_idx} has no valid output scale, using default")
                scale = 1.0
            
            # Convert to multiplier and shift
            mult, shift = quantization_params_to_multiplier_shift(scale)
            
            # Pack as 38-bit value: [37:32] shift, [31:0] multiplier
            entry = ((shift & 0x3F) << 32) | (mult & 0xFFFFFFFF)
            rom_data.append(entry)
            
            print(f"  Layer {layer_idx}: {layer['output_name']}")
            print(f"    Output scale: {scale:.6e} -> mult=0x{mult:08x}, shift={shift}")
            print(f"    Output channels: {layer['output_channels']}")
        else:
            # Fill remaining entries with default scale
            print(f"Warning: Layer {layer_idx} not found, using default scale")
            mult, shift = quantization_params_to_multiplier_shift(1.0)
            entry = ((shift & 0x3F) << 32) | (mult & 0xFFFFFFFF)
            rom_data.append(entry)
            print(f"  Layer {layer_idx}: default scale=1.0 -> mult=0x{mult:08x}, shift={shift}")
    
    # Write hex file
    print(f"\nWriting ROM data to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("// Requantize Scale ROM Data\n")
        f.write("// Generated from TensorFlow Lite model\n")
        f.write(f"// Source model: {model_path}\n")
        f.write("// Format: Each line is a 38-bit value (10 hex digits)\n") 
        f.write("// [37:32] shift (6 bits), [31:0] multiplier (32 bits)\n")
        f.write(f"// Total entries: {len(rom_data)}\n")
        f.write(f"// NUM_LAYERS: {NUM_LAYERS}\n\n")
        
        for i, entry in enumerate(rom_data):
            # Write as 10 hex digits (38 bits, padded to 40 bits)
            f.write(f"{entry:010x}\n")
    
    print(f"Successfully generated {output_file} with {len(rom_data)} entries")
    print(f"ROM depth: {len(rom_data)}, Data width: 38 bits")
    
    # Generate a summary
    print("\nROM Layout Summary:")
    for i in range(min(NUM_LAYERS, len(conv2d_layers))):
        layer_name = conv2d_layers[i]['output_name'] if i < len(conv2d_layers) else "default"
        print(f"  Address {i}: Layer {i} output scale ({layer_name})")

def main():
    parser = argparse.ArgumentParser(
        description="Generate requantize scale ROM hex file from TensorFlow Lite model"
    )
    parser.add_argument(
        "model_path",
        help="Path to the input TensorFlow Lite (.tflite) model file"
    )
    parser.add_argument(
        "-o", "--output",
        default="quant_params.hex",
        help="Output hex file path (default: quant_params.hex)"
    )
    
    args = parser.parse_args()
    
    if not args.model_path.endswith('.tflite'):
        print("Warning: Input file does not have .tflite extension")
    
    generate_requantize_rom_hex(args.model_path, args.output)

if __name__ == "__main__":
    main() 