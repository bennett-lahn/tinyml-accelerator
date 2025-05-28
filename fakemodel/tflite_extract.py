import tensorflow as tf
import numpy as np
import os

def extract_and_format_tflite_weights(tflite_model_path="simple_cnn_32x32_quant_int8.tflite"):
    """
    Loads a pre-existing .tflite model, extracts quantized weights and biases,
    and writes them to separate .hex files in Keras layer order.
    - Conv2D kernels: row-major, 16 values (32 hex chars) per line, no spaces. Scales/ZPs comments removed in their hex file.
    - Dense kernels: row-major, one hex byte per line. Scales/ZPs comments included.
    - Biases: int32, each 32-bit value on one line as 8 hex characters. Scales/ZPs comments included.
    """
    if not os.path.exists(tflite_model_path):
        print(f"Error: TFLite model file not found at {tflite_model_path}")
        return

    print(f"\n--- Loading TFLite Model from: {tflite_model_path} ---")
    tflite_int8_weights_for_rom = None 
    try:
        with open(tflite_model_path, 'rb') as f:
            tflite_quant_model_content = f.read()
        
        print("\n--- Inspecting TFLite Model for Quantized Weights and Parameters ---")
        interpreter = tf.lite.Interpreter(model_content=tflite_quant_model_content)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        tflite_int8_weights_for_rom = {} 

        for detail in tensor_details:
            name_lower = detail['name'].lower()
            is_a_qconst_tensor = name_lower.startswith("tfl.pseudo_qconst")

            if is_a_qconst_tensor and detail['quantization_parameters']['scales'].size > 0:
                tflite_int8_weights_for_rom[detail['name']] = {
                    'weights': interpreter.get_tensor(detail['index']), 
                    'original_shape': detail['shape'], 
                    'dtype': detail['dtype'], 
                    'scales': detail['quantization_parameters']['scales'],
                    'zero_points': detail['quantization_parameters']['zero_points']
                }
    except Exception as e:
        print(f"Error during TFLite model loading or inspection: {e}")
        return 

    if not tflite_int8_weights_for_rom:
        print("No constant tensors (weights/biases) were extracted from the TFLite model.")
        return

    print(f"\nSuccessfully extracted {len(tflite_int8_weights_for_rom)} constant 'tfl.pseudo_qconst' tensors.")

    ordered_tflite_tensor_names = [
        "tfl.pseudo_qconst11", # conv1 kernel
        "tfl.pseudo_qconst10", # conv1 bias
        "tfl.pseudo_qconst9",  # conv2 kernel
        "tfl.pseudo_qconst8",  # conv2 bias
        "tfl.pseudo_qconst7",  # conv3 kernel
        "tfl.pseudo_qconst6",  # conv3 bias
        "tfl.pseudo_qconst5",  # conv4 kernel
        "tfl.pseudo_qconst4",  # conv4 bias
        "tfl.pseudo_qconst3",  # dense1 kernel
        "tfl.pseudo_qconst2",  # dense1 bias
        "tfl.pseudo_qconst1",  # output_softmax kernel
        "tfl.pseudo_qconst"    # output_softmax bias
    ]

    missing_tensors = [name for name in ordered_tflite_tensor_names if name not in tflite_int8_weights_for_rom]
    if missing_tensors:
        print(f"Error: The following TFLite tensor names defined in 'ordered_tflite_tensor_names' were not found in the extracted tensors: {missing_tensors}")
        print("Please verify the TFLite tensor names and their mapping to Keras layers.")
        return

    print("\n--- Writing Extracted TFLite Weights and Biases to Separate .hex files (Keras Layer Order) ---")
    output_conv_kernels_hex_file = "tflite_conv_kernel_weights.hex"
    output_dense_kernels_hex_file = "tflite_dense_kernel_weights.hex"
    output_biases_hex_file = "tflite_bias_weights.hex"
    
    total_conv_kernel_bytes = 0
    total_dense_kernel_bytes = 0
    total_bias_bytes = 0
    
    with open(output_conv_kernels_hex_file, "w") as f_conv_k, \
         open(output_dense_kernels_hex_file, "w") as f_dense_k, \
         open(output_biases_hex_file, "w") as f_biases:
        
        f_conv_k.write("# Conv2D Kernel Weights (int8) - Keras Layer Order. Stored as 4x4 planes, row-major flattened, 16 values (32 hex chars) per line, no spaces. Scales/ZPs handled separately by hardware.\n")
        f_dense_k.write("# Dense Kernel Weights (int8) - Keras Layer Order. Stored row-major flattened, one hex byte per line\n")
        f_biases.write("# Bias Weights (Typically int32 from TFLite) - Keras Layer Order. Each 32-bit value on one line as 8 hex characters.\n") # Updated comment

        for tensor_name in ordered_tflite_tensor_names: 
            tensor_info = tflite_int8_weights_for_rom[tensor_name]
            weights_data = tensor_info['weights']
            original_shape = list(tensor_info['original_shape']) 
            data_type = tensor_info['dtype']
            
            if data_type == np.int8: 
                if len(original_shape) == 4 and original_shape[1] == 4 and original_shape[2] == 4: 
                    conv_kernel_header_comments = (
                        f"# Tensor Name (TFLite): {tensor_name}\n"
                        f"# Original Shape: {original_shape}\n"
                        f"# Dtype: {data_type}\n"
                    )
                    f_conv_k.write(conv_kernel_header_comments)
                    num_output_channels = original_shape[0]
                    num_input_channels = original_shape[3]
                    
                    for out_c in range(num_output_channels):
                        for in_c in range(num_input_channels):
                            kernel_plane = weights_data[out_c, :, :, in_c] 
                            if kernel_plane.shape != (4,4):
                                print(f"WARNING: Extracted plane for {tensor_name} (out_c={out_c}, in_c={in_c}) has unexpected shape {kernel_plane.shape}. Skipping.")
                                continue
                            
                            row_flattened_plane = kernel_plane.flatten(order='C') 
                            f_conv_k.write(f"# Keras Layer (derived): Conv Kernel Plane - OutputChannel={out_c}, InputChannel={in_c}\n")
                            
                            hex_chars_for_plane_line = []
                            for val_int8 in row_flattened_plane:
                                python_int_val = int(val_int8)
                                hex_val = format(python_int_val & 0xFF, '02x')
                                hex_chars_for_plane_line.append(hex_val)
                            
                            f_conv_k.write("".join(hex_chars_for_plane_line) + "\n") 
                            total_conv_kernel_bytes += len(row_flattened_plane)
                    f_conv_k.write("# End Tensor\n\n")
                elif len(original_shape) == 2: 
                    dense_kernel_header_comments = (
                        f"# Tensor Name (TFLite): {tensor_name}\n"
                        f"# Original Shape: {original_shape}\n"
                        f"# Dtype: {data_type}\n"
                        f"# Scales: {tensor_info['scales']}\n"
                        f"# Zero Points: {tensor_info['zero_points']}\n"
                    )
                    f_dense_k.write(dense_kernel_header_comments)
                    flattened_data = weights_data.flatten() 
                    for val_int8 in flattened_data:
                        python_int_val = int(val_int8)
                        hex_val = format(python_int_val & 0xFF, '02x')
                        f_dense_k.write(hex_val + "\n") 
                        total_dense_kernel_bytes += 1
                    f_dense_k.write("# End Tensor\n\n")
                else: 
                    generic_int8_header_comments = (
                        f"# Tensor Name (TFLite): {tensor_name}\n"
                        f"# Original Shape: {original_shape}\n"
                        f"# Dtype: {data_type}\n"
                        f"# Scales: {tensor_info['scales']}\n"
                        f"# Zero Points: {tensor_info['zero_points']}\n"
                    )
                    print(f"INFO: Generic int8 tensor '{tensor_name}' (shape {original_shape}) not categorized as Conv or Dense kernel. Writing to conv_kernels file by default (one hex byte per line).")
                    f_conv_k.write(generic_int8_header_comments) 
                    flattened_data = weights_data.flatten()
                    for val_int8 in flattened_data:
                        python_int_val = int(val_int8)
                        hex_val = format(python_int_val & 0xFF, '02x')
                        f_conv_k.write(hex_val + "\n")
                        total_conv_kernel_bytes += 1
                    f_conv_k.write("# End Tensor\n\n")

            elif data_type == np.int32: # Assumed to be Biases
                bias_header_comments = (
                    f"# Tensor Name (TFLite): {tensor_name}\n"
                    f"# Original Shape: {original_shape}\n"
                    f"# Dtype: {data_type}\n"
                    f"# Scales: {tensor_info['scales']}\n"
                    f"# Zero Points: {tensor_info['zero_points']}\n"
                )
                f_biases.write(bias_header_comments)
                flattened_data = weights_data.flatten()
                for val_int32 in flattened_data: # Each val_int32 is one bias value
                    try:
                        python_int_val = int(val_int32)
                        
                        # Format the 32-bit integer as an 8-character hex string
                        # (val & 0xFFFFFFFF) ensures correct representation for negative numbers in 2's complement
                        hex_32bit_value_string = format(python_int_val & 0xFFFFFFFF, '08x')
                        f_biases.write(hex_32bit_value_string + "\n")
                        total_bias_bytes += 4 # Each int32 bias value is 4 bytes
                                                
                    except Exception as e_bytes:
                        print(f"Error converting value {python_int_val} (from tensor {tensor_name}) to 32-bit hex: {e_bytes}. Skipping this value for bias file.")
                f_biases.write("# End Tensor\n\n")
            else:
                print(f"WARNING: Skipping tensor {tensor_name} for hex output due to unhandled dtype: {data_type}")

    print(f"\nExtracted TFLite Conv2D kernel weights written to {output_conv_kernels_hex_file}")
    print(f"Total individual hex values (bytes) for Conv2D kernels: {total_conv_kernel_bytes}")
    print(f"\nExtracted TFLite Dense kernel weights written to {output_dense_kernels_hex_file}")
    print(f"Total individual hex values (bytes) for Dense kernels: {total_dense_kernel_bytes}")
    print(f"\nExtracted TFLite bias weights written to {output_biases_hex_file}")
    print(f"Total individual hex values (bytes) for biases: {total_bias_bytes}")

if __name__ == '__main__':
    model_file_path = 'simple_cnn_32x32_quant_int8.tflite' 
    extract_and_format_tflite_weights(tflite_model_path=model_file_path)
