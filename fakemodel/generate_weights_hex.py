import sys
import tensorflow as tf
import numpy as np

# Global constants
KERNEL_SIZE = 4
VECTOR_WIDTH = 4  # For input channel grouping in regular layers

# Layer definitions based on the problem (Layers 0-3)
# Format: (IN_C, OUT_C, IS_SPECIAL_CASE)
# Based on CONV_IN_C = '{1,  8,  16,  32, ...}' and CONV_OUT_C = '{8,  16, 32,  64, ...}'
LAYER_CONFIGS = [
    (1, 8, True),   # Layer 0 (Special: 1 input channel, 8 output channels)
    (8, 16, False),  # Layer 1 (Regular: 8 input channels, 16 output channels)
    (16, 32, False), # Layer 2 (Regular: 16 input channels, 32 output channels)
    (32, 64, False)  # Layer 3 (Regular: 32 input channels, 64 output channels)
]

# --- Model Weight Storage ---
model_weights = {}  # Structure: model_weights[layer_idx][out_ch_idx][in_ch_idx][row_idx][col_idx]

def load_tflite_weights(model_path):
    """
    Loads weights from a TFLite model file into the global model_weights dictionary.
    Assumes the TFLite model's Conv2D layers (matching LAYER_CONFIGS) appear
    in the same order and have weights of dtype int8.
    """
    print(f"Loading TFLite model weights from: {model_path}")
    global model_weights # Ensure we're modifying the global dict
    model_weights = {}

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    except Exception as e:
        print(f"Error: Failed to load TFLite model from '{model_path}'. Ensure TensorFlow Lite is installed and the path is correct.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
        
    interpreter.allocate_tensors()
    all_tensor_details = interpreter.get_tensor_details()

    # Attempt to find and match weight tensors for Conv2D layers
    # This relies on the order of Conv2D weight tensors in the TFLite model
    # matching the order in LAYER_CONFIGS.
    
    # Filter for potential weight tensors: 4D, int8 dtype, and kernel size match.
    # This list will store the original indices from all_tensor_details
    candidate_weight_tensor_indices = []
    for i, detail in enumerate(all_tensor_details):
        shape = list(detail.get('shape', []))
        dtype = detail.get('dtype')
        
        if len(shape) == 4 and shape[1] == KERNEL_SIZE and shape[2] == KERNEL_SIZE:
            if dtype == np.int8: # Check for int8 type specifically
                candidate_weight_tensor_indices.append(i) # Store original index

    current_candidate_list_idx = 0 # Index into candidate_weight_tensor_indices

    for layer_idx, (expected_in_c, expected_out_c, _) in enumerate(LAYER_CONFIGS):
        print(f"  Processing Layer {layer_idx}: Expected IN_C={expected_in_c}, OUT_C={expected_out_c}")
        model_weights[layer_idx] = {}
        found_tensor_for_layer = False

        # Search for the next matching tensor in our filtered candidates
        temp_search_idx = current_candidate_list_idx
        while temp_search_idx < len(candidate_weight_tensor_indices):
            original_tensor_detail_idx = candidate_weight_tensor_indices[temp_search_idx]
            tensor_detail = all_tensor_details[original_tensor_detail_idx]
            
            t_shape = list(tensor_detail['shape']) # TFLite Conv2D weights: [out_c, H, W, in_c]
            t_out_c, t_h, t_w, t_in_c = t_shape[0], t_shape[1], t_shape[2], t_shape[3]

            if (t_out_c == expected_out_c and
                t_in_c == expected_in_c and
                t_h == KERNEL_SIZE and # Should already be true due to pre-filtering
                t_w == KERNEL_SIZE):   # Should already be true
                
                print(f"    Found matching weight tensor: {tensor_detail['name']} (original index {original_tensor_detail_idx}) with shape {t_shape}")
                weights_data = interpreter.get_tensor(tensor_detail['index']) # Use original index here
                
                # Populate model_weights: model_weights[layer_idx][o][i][r_k][c_k]
                # TFLite weights_data shape: [expected_out_c, KERNEL_SIZE, KERNEL_SIZE, expected_in_c]
                for o in range(expected_out_c):
                    model_weights[layer_idx][o] = {}
                    for i in range(expected_in_c):
                        model_weights[layer_idx][o][i] = {}
                        for r_k in range(KERNEL_SIZE):
                            model_weights[layer_idx][o][i][r_k] = {}
                            for c_k in range(KERNEL_SIZE):
                                # TFLite tensor access: [out_channel, height, width, in_channel]
                                weight_val = weights_data[o, r_k, c_k, i]
                                model_weights[layer_idx][o][i][r_k][c_k] = int(weight_val) # Ensure Python int
                
                current_candidate_list_idx = temp_search_idx + 1 # Consume this candidate
                found_tensor_for_layer = True
                break # Found tensor for this layer_idx, move to next layer_idx
            
            temp_search_idx += 1

        if not found_tensor_for_layer:
            print(f"Error: Could not find a matching TFLite weight tensor for Layer {layer_idx} with IN_C={expected_in_c}, OUT_C={expected_out_c}, KERNEL_SIZE={KERNEL_SIZE}x{KERNEL_SIZE} and dtype int8.", file=sys.stderr)
            print("Please ensure the TFLite model contains the expected layers in the correct order and that they are quantized to int8.", file=sys.stderr)
            print(f"Searched {len(candidate_weight_tensor_indices)} potential int8 weight tensors with KERNEL_SIZE={KERNEL_SIZE}.", file=sys.stderr)
            sys.exit(1)

    print("TFLite model weights loaded successfully into model_weights.")

# --- Helper for 8-bit signed int to 2-char hex ---
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

# --- Word Packing Functions ---

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
    # Kernel is 4x4 (input channel is implicitly 0 for layer 0)
    # Access weights: model_weights[layer_idx][out_ch_idx][0][row_k][col_k]

    # word_parts_hex stores [data0_hex, data1_hex, data2_hex, data3_hex]
    # data0 is MSB part of 128-bit word, data3 is LSB part.
    word_parts_hex = [""] * 4  # Index 0 for data0, Index 3 for data3

    for r_k in range(KERNEL_SIZE):  # Iterate through kernel rows 0, 1, 2, 3
        row_data_hex_segments = []  # To store hex weights for columns [Col0, Col1, Col2, Col3]
        for c_k in range(KERNEL_SIZE):  # Iterate through kernel columns 0, 1, 2, 3
            weight = model_weights[layer_idx][out_ch_idx][0][r_k][c_k]
            row_data_hex_segments.append(s8_to_hex(weight))

        # Assemble the 32-bit hex string for the current row: W(r,C3)W(r,C2)W(r,C1)W(r,C0)
        # row_data_hex_segments is [hex(C0), hex(C1), hex(C2), hex(C3)]
        # We need to reverse for MSB first: hex(C3) + hex(C2) + hex(C1) + hex(C0)
        current_row_hex = "".join(reversed(row_data_hex_segments))

        # Assign to the correct part of the 128-bit word based on row index
        # data3 is Row 0, data2 is Row 1, data1 is Row 2, data0 is Row 3
        if r_k == 0:   # Kernel Row 0
            word_parts_hex[3] = current_row_hex  # Goes into data3
        elif r_k == 1: # Kernel Row 1
            word_parts_hex[2] = current_row_hex  # Goes into data2
        elif r_k == 2: # Kernel Row 2
            word_parts_hex[1] = current_row_hex  # Goes into data1
        elif r_k == 3: # Kernel Row 3
            word_parts_hex[0] = current_row_hex  # Goes into data0

    # Combine all parts: data0_hex | data1_hex | data2_hex | data3_hex
    # This forms the 128-bit word with data0 as MSB.
    final_hex_word = "".join(word_parts_hex)
    if len(final_hex_word) != 32: # 128 bits / 4 bits_per_hex_char = 32 chars
        raise ValueError(f"Layer 0: Generated hex word '{final_hex_word}' for L{layer_idx} O{out_ch_idx} is not 32 characters long.")
    return final_hex_word

def pack_regular_layer_row_to_word_hex(layer_idx, out_ch_idx, in_ch_group_start, r_k):
    """
    Packs one row of a 4x4 kernel (across 4 input channels) for regular layers
    into a 128-bit hex string.
    Regular Layers (1, 2, 3): One 128-bit word per kernel row, for a group of 4 input channels.
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
    # r_k is the current kernel row (0-3)
    # in_ch_group_start is the starting input channel for this group (e.g., 0, 4, 8...)

    # word_parts_hex stores [data0_hex, data1_hex, data2_hex, data3_hex]
    word_parts_hex = [""] * 4  # Index 0 for data0 (Col3), Index 3 for data3 (Col0)

    for c_k in range(KERNEL_SIZE):  # Iterate through kernel columns 0, 1, 2, 3
        col_data_hex_segments = []  # To store hex weights for input channels [IC0, IC1, IC2, IC3]
        for i_offset in range(VECTOR_WIDTH):  # Iterate input channels within the group 0, 1, 2, 3
            in_ch_abs = in_ch_group_start + i_offset  # Absolute input channel index
            weight = model_weights[layer_idx][out_ch_idx][in_ch_abs][r_k][c_k]
            col_data_hex_segments.append(s8_to_hex(weight))

        # Assemble 32-bit hex string for current column: W(ic3)W(ic2)W(ic1)W(ic0)
        # col_data_hex_segments is [hex(IC0), hex(IC1), hex(IC2), hex(IC3)]
        # We need to reverse for MSB first: hex(IC3) + hex(IC2) + hex(IC1) + hex(IC0)
        current_col_data_hex = "".join(reversed(col_data_hex_segments))

        # Assign to the correct part of the 128-bit word based on column index
        # data3 is Col 0, data2 is Col 1, data1 is Col 2, data0 is Col 3
        if c_k == 0:   # Kernel Col 0
            word_parts_hex[3] = current_col_data_hex # Goes into data3
        elif c_k == 1: # Kernel Col 1
            word_parts_hex[2] = current_col_data_hex # Goes into data2
        elif c_k == 2: # Kernel Col 2
            word_parts_hex[1] = current_col_data_hex # Goes into data1
        elif c_k == 3: # Kernel Col 3
            word_parts_hex[0] = current_col_data_hex # Goes into data0

    # Combine all parts: data0_hex | data1_hex | data2_hex | data3_hex
    final_hex_word = "".join(word_parts_hex)
    if len(final_hex_word) != 32:
        raise ValueError(f"Regular Layer: Generated hex word '{final_hex_word}' for L{layer_idx} O{out_ch_idx} ICG{in_ch_group_start//VECTOR_WIDTH} R{r_k} is not 32 characters long.")
    return final_hex_word

# --- Main script logic ---
def generate_hex_file(output_filename="initial_weights.hex"):
    """
    Generates the ROM initialization hex file.
    """
    # Load weights from the TFLite model
    # Ensure the 'fakemodel' directory exists and 'model.tflite' is inside it.
    load_tflite_weights("simple_cnn_32x32_quant_int8.tflite")
    
    all_rom_words_hex = []
    current_address = 0

    print("\\nProcessing layers and generating ROM words...")

    # Layer 0 (Special Case)
    # Addressing: Words are laid out one per output channel.
    l0_cfg_idx = 0
    l0_in_c, l0_out_c, _ = LAYER_CONFIGS[l0_cfg_idx]
    print(f"Layer {l0_cfg_idx} (Special): IN_C={l0_in_c}, OUT_C={l0_out_c}")
    for o_idx in range(l0_out_c):  # Iterate output channels
        # Each output channel's 4x4x1 kernel forms one ROM word
        hex_word = pack_layer0_kernel_to_word_hex(l0_cfg_idx, o_idx)
        all_rom_words_hex.append(hex_word)
        # print(f"  L0 Addr {current_address:04x}: O_CH={o_idx}, Word={hex_word}")
        current_address += 1
    print(f"Layer {l0_cfg_idx} processed. Words generated: {l0_out_c}")

    # Regular Layers (1, 2, 3)
    # Addressing: Layer -> Output Channel -> Input Channel Group -> Kernel Row
    for layer_config_list_idx, (in_c, out_c, _) in enumerate(LAYER_CONFIGS[1:]):
        actual_layer_idx = layer_config_list_idx + 1 # Actual layer index (1, 2, or 3)
        
        print(f"Layer {actual_layer_idx} (Regular): IN_C={in_c}, OUT_C={out_c}")
        if in_c % VECTOR_WIDTH != 0:
            # This check should ideally be handled by the TFLite loader finding a mismatch,
            # but good to keep as a safeguard for LAYER_CONFIGS.
            raise ValueError(f"Layer {actual_layer_idx} IN_C ({in_c}) is not divisible by VECTOR_WIDTH ({VECTOR_WIDTH}) for ROM packing.")
        num_input_channel_groups = in_c // VECTOR_WIDTH
        
        layer_words_count = 0
        for o_idx in range(out_c):  # Iterate output channels
            for group_idx in range(num_input_channel_groups):  # Iterate input channel groups
                in_ch_group_start = group_idx * VECTOR_WIDTH
                # For each (output_channel, input_channel_group), a 4x4x4 kernel slice is processed.
                # This slice results in KERNEL_SIZE (4) ROM words, one for each row.
                for r_k in range(KERNEL_SIZE):  # Iterate kernel rows (0, 1, 2, 3)
                    hex_word = pack_regular_layer_row_to_word_hex(actual_layer_idx, o_idx, in_ch_group_start, r_k)
                    all_rom_words_hex.append(hex_word)
                    # print(f"  L{actual_layer_idx} Addr {current_address:04x}: O_CH={o_idx}, ICG={group_idx}, Row={r_k}, Word={hex_word}")
                    current_address += 1
                    layer_words_count +=1
        print(f"Layer {actual_layer_idx} processed. Words generated: {layer_words_count}")


    # Write to file
    try:
        with open(output_filename, "w") as f:
            for hex_word in all_rom_words_hex:
                f.write(hex_word + "\\n")
        print(f"\\nSuccessfully generated ROM initialization file: {output_filename}")
        print(f"Total ROM words written: {len(all_rom_words_hex)}")
        print(f"Next available ROM address: {current_address:04x}")
    except IOError as e:
        print(f"Error writing to file {output_filename}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    generate_hex_file("initial_weights.hex") 