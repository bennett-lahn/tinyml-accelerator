import sys
import tensorflow as tf
import numpy as np

# Global constants
KERNEL_SIZE = 4
VECTOR_WIDTH = 4  # For input channel grouping in regular layers

# Layer definitions based on the problem (Layers 0-3)
# Format: (IN_C, OUT_C, IS_SPECIAL_CASE)
# For Layer 0, IN_C will be forced to 1. For other layers, IN_C = -1 infers from previous OUT_C.
LAYER_CONFIGS = [
    (1, 8, True),   # Layer 0 (Special: 1 input channel, 8 output channels)
    (8, 16, False),  # Layer 1 (Regular: 8 input channels, 16 output channels)
    (16, 32, False), # Layer 2 (Regular: 16 input channels, 32 output channels)
    (32, 64, False)  # Layer 3 (Regular: 32 input channels, 64 output channels)
]

# --- Model Weight Storage ---
model_weights = {}  # Structure: model_weights[layer_idx][out_ch_idx][in_ch_idx][row_idx][col_idx]

def load_tflite_conv_weights(model_path):
    """
    Loads Conv2D weights from a TFLite model using the tfl.pseudo_qconst tensor approach.
    Populates the global model_weights dictionary and returns processed layer configurations.
    """
    print(f"Loading TFLite model Conv2D weights from: {model_path}")
    global model_weights 
    model_weights = {}

    try:
        with open(model_path, 'rb') as f:
            tflite_model_content = f.read()
        
        interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        
    except Exception as e:
        print(f"Error: Failed to load TFLite model from '{model_path}'. \nEnsure TensorFlow Lite is installed and the path is correct.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract Conv2D weight tensors (tfl.pseudo_qconst with 4D shape and int8 dtype)
    conv_weight_tensors = {}
    
    for detail in tensor_details:
        name_lower = detail['name'].lower()
        is_qconst_tensor = name_lower.startswith("tfl.pseudo_qconst")
        
        if is_qconst_tensor and detail['dtype'] == np.int8:
            shape = list(detail['shape'])
            if len(shape) == 4 and shape[1] == KERNEL_SIZE and shape[2] == KERNEL_SIZE:
                weights_data = interpreter.get_tensor(detail['index'])
                conv_weight_tensors[detail['name']] = {
                    'weights': weights_data,
                    'shape': shape,
                    'name': detail['name']
                }
                print(f"  Found Conv2D weight tensor: {detail['name']} with shape {shape}")

    if not conv_weight_tensors:
        print("Error: No Conv2D weight tensors found in TFLite model.", file=sys.stderr)
        sys.exit(1)

    # Order the Conv2D tensors by their pseudo_qconst number (descending to get layer order)
    ordered_conv_tensor_names = [
        "tfl.pseudo_qconst11", # conv1 kernel
        "tfl.pseudo_qconst9",  # conv2 kernel  
        "tfl.pseudo_qconst7",  # conv3 kernel
        "tfl.pseudo_qconst5",  # conv4 kernel
    ]
    
    # Filter to only include tensors that exist and are Conv2D
    available_conv_tensors = [name for name in ordered_conv_tensor_names if name in conv_weight_tensors]
    
    if len(available_conv_tensors) < len(LAYER_CONFIGS):
        print(f"Error: Found only {len(available_conv_tensors)} Conv2D tensors, but LAYER_CONFIGS expects {len(LAYER_CONFIGS)} layers.", file=sys.stderr)
        sys.exit(1)

    # --- Determine Processed Layer Configurations (with inferred IN_C) --- 
    processed_layer_configs = []
    for layer_idx, (in_c_config, out_c_config, is_special) in enumerate(LAYER_CONFIGS):
        actual_in_c = in_c_config
        if layer_idx == 0:
            if in_c_config == -1:
                print("Error: IN_C cannot be -1 for Layer 0.", file=sys.stderr)
                sys.exit(1)
            actual_in_c = 1 # Force Layer 0 IN_C to 1
            if in_c_config != 1 and in_c_config != -1:
                 print(f"  Info: Layer 0 IN_C from LAYER_CONFIGS ({in_c_config}) overridden to 1.")
        elif in_c_config == -1: 
            if not processed_layer_configs: 
                print(f"Error: Cannot infer IN_C for Layer {layer_idx} due to missing previous layer processed_config.", file=sys.stderr)
                sys.exit(1)
            actual_in_c = processed_layer_configs[layer_idx - 1][1] 
            print(f"  Info: Layer {layer_idx} IN_C specified as -1, inferred as {actual_in_c} from Layer {layer_idx-1}'s OUT_C.")
        processed_layer_configs.append((actual_in_c, out_c_config, is_special))

    # --- Populate model_weights from TFLite Conv2D tensors ---
    for layer_idx, (expected_in_c, expected_out_c, _) in enumerate(processed_layer_configs):
        print(f"  Processing Layer {layer_idx}: Using IN_C={expected_in_c}, OUT_C={expected_out_c} from LAYER_CONFIGS.")
        model_weights[layer_idx] = {}

        tensor_name = available_conv_tensors[layer_idx]
        tensor_info = conv_weight_tensors[tensor_name]
        weights_data = tensor_info['weights']
        
        tflite_actual_out_c, _, _, tflite_actual_in_c = weights_data.shape
        print(f"    Mapping to TFLite Conv2D weight tensor: {tensor_name} with actual shape [{tflite_actual_out_c}, {KERNEL_SIZE}, {KERNEL_SIZE}, {tflite_actual_in_c}]")

        if expected_out_c > tflite_actual_out_c or expected_in_c > tflite_actual_in_c:
            print(f"Error: For Layer {layer_idx}, LAYER_CONFIGS expects dimensions (OUT_C={expected_out_c}, IN_C={expected_in_c})", file=sys.stderr)
            print(f"       but the corresponding TFLite tensor '{tensor_name}' only has dimensions (OUT_C={tflite_actual_out_c}, IN_C={tflite_actual_in_c}).", file=sys.stderr)
            print("       Cannot read out of bounds from the TFLite tensor.", file=sys.stderr)
            sys.exit(1)
        
        if expected_out_c < tflite_actual_out_c or expected_in_c < tflite_actual_in_c:
            print(f"  Warning: For Layer {layer_idx}, LAYER_CONFIGS dimensions (OUT_C={expected_out_c}, IN_C={expected_in_c})", file=sys.stderr)
            print(f"           are smaller than the TFLite tensor '{tensor_name}' (OUT_C={tflite_actual_out_c}, IN_C={tflite_actual_in_c}).", file=sys.stderr)
            print("           A subset of the TFLite tensor will be used.", file=sys.stderr)

        for o in range(expected_out_c):
            model_weights[layer_idx][o] = {}
            for i in range(expected_in_c):
                model_weights[layer_idx][o][i] = {}
                for r_k in range(KERNEL_SIZE):
                    model_weights[layer_idx][o][i][r_k] = {}
                    for c_k in range(KERNEL_SIZE):
                        weight_val = weights_data[o, r_k, c_k, i]
                        model_weights[layer_idx][o][i][r_k][c_k] = int(weight_val)
                
    print("TFLite Conv2D weights loaded successfully into model_weights based on LAYER_CONFIGS.")
    return processed_layer_configs

# --- Helper for 8-bit signed int to 2-char hex (copied exactly from generate_weights_hex.py) ---
def s8_to_hex(val):
    """
    Converts an 8-bit signed integer to its 2's complement 2-character hex representation.
    Example: 1 -> "01", -1 -> "ff", -128 -> "80"
    """
    if not -128 <= val <= 127:
        raise ValueError(f"Value {val} is out of 8-bit signed range [-128, 127]")
    if val < 0:
        val = 256 + val  # 2's complement for negative numbers
    return format(val, '02x')

# --- Word Packing Functions (copied exactly from generate_weights_hex.py) ---

def pack_layer0_kernel_to_word_hex(layer_idx, out_ch_idx):
    """
    Packs a 4x4x1 kernel for Layer 0 into a 128-bit hex string.
    Layer 0 (Special Case): One 128-bit word per output filter.
    ROM Word Structure (128 bits total, MSB first in hex string):
        data0 (bits 127:96) : Kernel Row 3
        data1 (bits 95:64)  : Kernel Row 2
        data2 (bits 63:32)  : Kernel Row 1
        data3 (bits 31:0)   : Kernel Row 0 (LSB part of 128-bit word)

    Each 32-bit dataX segment (representing one kernel row):
        MSB bits 31:24 : Weight for (current_row, Col 3)
        bits 23:16     : Weight for (current_row, Col 2)
        bits 15:8      : Weight for (current_row, Col 1)
        LSB bits 7:0   : Weight for (current_row, Col 0)
    """
    # Access weights: model_weights[layer_idx][out_ch_idx][0][row_k][col_k]
    word_parts_hex = [""] * 4  

    for r_k in range(KERNEL_SIZE):  
        row_data_hex_segments = []  
        for c_k in range(KERNEL_SIZE):  
            weight = model_weights[layer_idx][out_ch_idx][0][r_k][c_k]
            row_data_hex_segments.append(s8_to_hex(weight))

        current_row_hex = "".join(reversed(row_data_hex_segments))

        if r_k == 0:   
            word_parts_hex[3] = current_row_hex  
        elif r_k == 1: 
            word_parts_hex[2] = current_row_hex  
        elif r_k == 2: 
            word_parts_hex[1] = current_row_hex  
        elif r_k == 3: 
            word_parts_hex[0] = current_row_hex  

    final_hex_word = "".join(word_parts_hex)
    if len(final_hex_word) != 32: 
        raise ValueError(f"Layer 0: Generated hex word '{final_hex_word}' for L{layer_idx} O{out_ch_idx} is not 32 characters long.")
    return final_hex_word

def pack_regular_layer_row_to_word_hex(layer_idx, out_ch_idx, in_ch_group_start, r_k):
    """
    Packs one row of a 4x4 kernel (across 4 input channels) for regular layers
    into a 128-bit hex string.
    ROM Word Structure (128 bits total, MSB first in hex string):
        data0 (bits 127:96) : Weights for (current_row, Col 3) across 4 input channels
        data1 (bits 95:64)  : Weights for (current_row, Col 2) across 4 input channels
        data2 (bits 63:32)  : Weights for (current_row, Col 1) across 4 input channels
        data3 (bits 31:0)   : Weights for (current_row, Col 0) across 4 input channels (LSB part)

    Each 32-bit dataX segment (representing one kernel column 'c_k' for the current row 'r_k'):
        MSB bits 31:24 : Weight for (r_k, c_k, in_channel_3_of_group)
        bits 23:16     : Weight for (r_k, c_k, in_channel_2_of_group)
        bits 15:8      : Weight for (r_k, c_k, in_channel_1_of_group)
        LSB bits 7:0   : Weight for (r_k, c_k, in_channel_0_of_group)
    """
    word_parts_hex = [""] * 4  

    for c_k in range(KERNEL_SIZE):  
        col_data_hex_segments = []  
        for i_offset in range(VECTOR_WIDTH):  
            in_ch_abs = in_ch_group_start + i_offset  
            weight = model_weights[layer_idx][out_ch_idx][in_ch_abs][r_k][c_k]
            col_data_hex_segments.append(s8_to_hex(weight))

        current_col_data_hex = "".join(reversed(col_data_hex_segments))

        if c_k == 0:   
            word_parts_hex[3] = current_col_data_hex 
        elif c_k == 1: 
            word_parts_hex[2] = current_col_data_hex 
        elif c_k == 2: 
            word_parts_hex[1] = current_col_data_hex 
        elif c_k == 3: 
            word_parts_hex[0] = current_col_data_hex 

    final_hex_word = "".join(word_parts_hex)
    if len(final_hex_word) != 32:
        raise ValueError(f"Regular Layer: Generated hex word '{final_hex_word}' for L{layer_idx} O{out_ch_idx} ICG{in_ch_group_start//VECTOR_WIDTH} R{r_k} is not 32 characters long.")
    return final_hex_word

# --- Main script logic ---
def generate_hex_file(output_filename="conv_weights.hex"):
    """
    Generates the ROM initialization hex file for Conv2D weights only.
    """
    final_layer_configs = load_tflite_conv_weights("simple_cnn_32x32_quant_int8.tflite")
    
    all_rom_words_hex = []
    current_address = 0

    print("\nProcessing layers and generating ROM words using final configurations from LAYER_CONFIGS...")

    # Layer 0 (Special Case)
    l0_cfg_idx = 0
    l0_in_c, l0_out_c, _ = final_layer_configs[l0_cfg_idx]
    
    print(f"Layer {l0_cfg_idx} (Special): IN_C={l0_in_c}, OUT_C={l0_out_c}")
    for o_idx in range(l0_out_c):
        hex_word = pack_layer0_kernel_to_word_hex(l0_cfg_idx, o_idx)
        all_rom_words_hex.append(hex_word)
        current_address += 1
    print(f"Layer {l0_cfg_idx} processed. Words generated: {l0_out_c}")

    # Regular Layers (subsequent layers)
    for layer_config_list_idx, (current_layer_actual_in_c, current_layer_actual_out_c, _) in enumerate(final_layer_configs[1:]):
        actual_layer_idx = layer_config_list_idx + 1 
        
        print(f"Layer {actual_layer_idx} (Regular): IN_C={current_layer_actual_in_c}, OUT_C={current_layer_actual_out_c}")
        if current_layer_actual_in_c % VECTOR_WIDTH != 0:
            raise ValueError(f"Layer {actual_layer_idx} IN_C ({current_layer_actual_in_c}) is not divisible by VECTOR_WIDTH ({VECTOR_WIDTH}) for ROM packing.")
        num_input_channel_groups = current_layer_actual_in_c // VECTOR_WIDTH
        
        layer_words_count = 0
        for o_idx in range(current_layer_actual_out_c):
            for group_idx in range(num_input_channel_groups):
                in_ch_group_start = group_idx * VECTOR_WIDTH
                for r_k in range(KERNEL_SIZE):
                    hex_word = pack_regular_layer_row_to_word_hex(actual_layer_idx, o_idx, in_ch_group_start, r_k)
                    all_rom_words_hex.append(hex_word)
                    current_address += 1
                    layer_words_count +=1
        print(f"Layer {actual_layer_idx} processed. Words generated: {layer_words_count}")

    # Write to file
    try:
        with open(output_filename, "w") as f:
            for hex_word in all_rom_words_hex:
                f.write(hex_word + "\n")
        print(f"\nSuccessfully generated ROM initialization file: {output_filename}")
        print(f"Total ROM words written: {len(all_rom_words_hex)}")
        print(f"Next available ROM address: {current_address:04x}")
    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    generate_hex_file("conv_weights.hex") 