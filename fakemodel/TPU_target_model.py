import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os # For checking file existence

# --- 1. Model Definition (with 4x4 kernels and updated input shape) ---
def create_simple_int8_target_cnn(input_shape=(32, 32, 1), num_classes=10): # Default input_shape changed
    """
    Creates a small 4-layer CNN model suitable as a target for int8 TPU design,
    using 4x4 kernels and 32x32 input.
    """
    model = models.Sequential(name="Simple_CNN_32x32_for_int8_TPU_4x4_kernels") # Renamed for clarity
    model.add(layers.Conv2D(filters=8, kernel_size=(4, 4), activation=tf.nn.relu6, padding='same', input_shape=input_shape, name="conv1_relu6_4x4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1")) # Output: 16x16x8
    model.add(layers.Conv2D(filters=16, kernel_size=(4, 4), activation=tf.nn.relu6, padding='same', name="conv2_relu6_4x4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2")) # Output: 8x8x16
    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation=tf.nn.relu6, padding='same', name="conv3_relu6_4x4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3")) # Output: 4x4x32
    model.add(layers.Conv2D(filters=64, kernel_size=(4, 4), activation=tf.nn.relu6, padding='same', name="conv4_relu6_4x4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4")) # Output: 2x2x64
    model.add(layers.Flatten(name="flatten")) # Flattened shape: 2*2*64 = 256
    model.add(layers.Dense(units=64, activation=tf.nn.relu6, name="dense1_relu6"))
    model.add(layers.Dense(units=num_classes, activation='softmax', name="output_softmax"))
    return model

# --- 2. Train the Model (or load pre-trained weights) ---
INPUT_SHAPE = (32, 32, 1) # Grayscale 32x32
NUM_CLASSES = 10
model = create_simple_int8_target_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use sparse if labels are integers
              metrics=['accuracy'])

# --- !!! CHOOSE YOUR TRAINING DATA SOURCE !!! ---
USE_DUMMY_DATA = False # Set to True to use dummy data for very quick tests

if USE_DUMMY_DATA:
    print("\n--- Training Model (with dummy data for demonstration) ---")
    num_dummy_samples = 100
    dummy_x_train = np.random.rand(num_dummy_samples, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]).astype(np.float32)
    dummy_y_train = np.random.randint(0, NUM_CLASSES, size=(num_dummy_samples,)).astype(np.int32)
    representative_data_source = dummy_x_train
    model.fit(dummy_x_train, dummy_y_train, epochs=1, batch_size=10, verbose=1)
    print("--- Model Training Complete (dummy) ---")
else:
    print("\n--- Loading and Preparing Real Training Data (CIFAR-10 Grayscale 32x32) ---")
    cifar10 = tf.keras.datasets.cifar10
    (x_train_orig_cifar, y_train_orig_cifar), (x_test_orig_cifar, y_test_orig_cifar) = cifar10.load_data()

    # Preprocess CIFAR-10 data
    # 1. Convert to Grayscale (CIFAR-10 images are 32x32x3)
    x_train_gray_cifar = tf.image.rgb_to_grayscale(x_train_orig_cifar) # Output: (num_samples, 32, 32, 1), dtype=tf.uint8
    x_test_gray_cifar = tf.image.rgb_to_grayscale(x_test_orig_cifar)   # Output: (num_samples, 32, 32, 1), dtype=tf.uint8

    # 2. Normalize pixel values from [0, 255] to [0.0, 1.0] and ensure float32
    #    Cast to float32 BEFORE division
    x_train_float_cifar = tf.cast(x_train_gray_cifar, tf.float32)
    x_test_float_cifar = tf.cast(x_test_gray_cifar, tf.float32)

    x_train_processed_cifar = np.array(x_train_float_cifar / 255.0, dtype=np.float32)
    x_test_processed_cifar = np.array(x_test_float_cifar / 255.0, dtype=np.float32)
    
    representative_data_source = x_train_processed_cifar 

    y_train_processed_cifar = y_train_orig_cifar.flatten().astype(np.int32)
    y_test_processed_cifar = y_test_orig_cifar.flatten().astype(np.int32)

    print(f"x_train_processed_cifar shape: {x_train_processed_cifar.shape}, dtype: {x_train_processed_cifar.dtype}")
    print(f"y_train_processed_cifar shape: {y_train_processed_cifar.shape}, dtype: {y_train_processed_cifar.dtype}")

    print("\n--- Training Model (with CIFAR-10 Grayscale 32x32 data) ---")
    model.fit(x_train_processed_cifar, y_train_processed_cifar,
              epochs=10, 
              batch_size=32,
              validation_data=(x_test_processed_cifar, y_test_processed_cifar),
              verbose=1)
    print("--- Model Training Complete (CIFAR-10 Grayscale 32x32) ---")

# --- 3. TensorFlow Lite Conversion and Quantization ---
print("\n--- Step 3: TensorFlow Lite Conversion and Quantization ---")

def representative_dataset_gen():
  num_calibration_samples = 100
  for i in range(num_calibration_samples):
    # Ensure the sample is float32, as expected by the converter for calibration
    sample = representative_data_source[i:i+1].astype(np.float32) 
    yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Starting TFLite conversion and quantization (this may take a moment)...")
tflite_int8_weights_for_rom = None # Initialize in case of error
try:
    tflite_quant_model_content = converter.convert()
    print("TFLite conversion and quantization successful.")
    tflite_model_path = 'simple_cnn_32x32_quant_int8.tflite'
    with open(tflite_model_path, 'wb') as f:
      f.write(tflite_quant_model_content)
    print(f"Quantized INT8 TFLite model saved to: {tflite_model_path}")

    print("\n--- Inspecting TFLite Model for Quantized Weights and Parameters ---")
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model_content)
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    tflite_int8_weights_for_rom = {} # Initialize the dictionary

    for detail in tensor_details:
        name_lower = detail['name'].lower()
        is_a_qconst_tensor = name_lower.startswith("tfl.pseudo_qconst")

        if is_a_qconst_tensor and detail['quantization_parameters']['scales'].size > 0:
            print(f"Extracting Matching Constant Tensor: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Dtype: {detail['dtype']}") 
            print(f"  Quantization Scales: {detail['quantization_parameters']['scales']}")
            print(f"  Quantization Zero Points: {detail['quantization_parameters']['zero_points']}")
            
            tensor_data = interpreter.get_tensor(detail['index'])
            if tensor_data is None:
                print(f"  WARNING: Tensor data for {detail['name']} is None despite matching criteria. Skipping this tensor.")
                continue
            
            print(f"  Data (first 5 elements if available): {tensor_data.flatten()[:5]}")
            print("-" * 30)

            tflite_int8_weights_for_rom[detail['name']] = {
                'weights': tensor_data, 
                'original_shape': detail['shape'], 
                'dtype': detail['dtype'], 
                'scales': detail['quantization_parameters']['scales'],
                'zero_points': detail['quantization_parameters']['zero_points']
            }
except Exception as e:
    print(f"Error during TFLite conversion or inspection: {e}")

print("\n--- Reminder for TFLite Weights ---")
if tflite_int8_weights_for_rom is not None and tflite_int8_weights_for_rom:
    print(f"Successfully extracted {len(tflite_int8_weights_for_rom)} constant tensors (weights/biases).")
    print("The `tflite_int8_weights_for_rom` dictionary contains these quantized tensors and their parameters.")
else:
    print("No TFLite constant tensors (weights/biases) were extracted or an error occurred.")


# --- 6. MODIFIED: Write Extracted TFLite Weights and Biases to Separate .hex files ---
if tflite_int8_weights_for_rom: 
    print("\n--- Step 6: Writing Extracted TFLite Weights and Biases to Separate .hex files ---")
    output_conv_kernels_hex_file = "tflite_conv_kernel_weights.hex"
    output_dense_kernels_hex_file = "tflite_dense_kernel_weights.hex"
    output_biases_hex_file = "tflite_bias_weights.hex"
    
    total_conv_kernel_bytes = 0
    total_dense_kernel_bytes = 0
    total_bias_bytes = 0
    
    sorted_tensor_names = sorted(tflite_int8_weights_for_rom.keys())

    with open(output_conv_kernels_hex_file, "w") as f_conv_k, \
         open(output_dense_kernels_hex_file, "w") as f_dense_k, \
         open(output_biases_hex_file, "w") as f_biases:
        
        f_conv_k.write("# Conv2D Kernel Weights (int8) - Stored as 4x4 planes, column-major flattened\n")
        f_dense_k.write("# Dense Kernel Weights (int8) - Stored row-major flattened\n")
        f_biases.write("# Bias Weights (Typically int32 from TFLite, written as bytes)\n")

        for tensor_name in sorted_tensor_names:
            tensor_info = tflite_int8_weights_for_rom[tensor_name]
            weights_data = tensor_info['weights']
            original_shape = tensor_info['original_shape']
            data_type = tensor_info['dtype']
            
            header_comments = (
                f"# Tensor Name: {tensor_name}\n"
                f"# Original Shape: {original_shape}\n"
                f"# Dtype: {data_type}\n"
                f"# Scales: {tensor_info['scales']}\n"
                f"# Zero Points: {tensor_info['zero_points']}\n"
            )
            
            if data_type == np.int8: # Kernels/Weights
                # Differentiate between Conv2D and Dense kernels based on shape
                if len(original_shape) == 4 and original_shape[1] == 4 and original_shape[2] == 4: # Conv2D 4x4 kernel
                    f_conv_k.write(header_comments)
                    num_output_channels = original_shape[0]
                    num_input_channels = original_shape[3]
                    kernel_h, kernel_w = original_shape[1], original_shape[2]

                    f_conv_k.write(f"# Processing as Conv2D Kernel. Output Channels: {num_output_channels}, Input Channels: {num_input_channels}, H: {kernel_h}, W: {kernel_w}\n")
                    for out_c in range(num_output_channels):
                        for in_c in range(num_input_channels):
                            kernel_plane = weights_data[out_c, :, :, in_c] 
                            if kernel_plane.shape != (4,4):
                                print(f"WARNING: Extracted plane for {tensor_name} (out_c={out_c}, in_c={in_c}) has unexpected shape {kernel_plane.shape}. Skipping.")
                                continue
                            column_flattened_plane = kernel_plane.flatten(order='F')
                            f_conv_k.write(f"# Kernel Plane: OutputChannel={out_c}, InputChannel={in_c}\n")
                            for val_int8 in column_flattened_plane:
                                # Ensure val_int8 is treated as a Python int for formatting
                                python_int_val = int(val_int8)
                                hex_val = format(python_int_val & 0xFF, '02x')
                                f_conv_k.write(hex_val + "\n")
                                total_conv_kernel_bytes += 1
                    f_conv_k.write("# End Tensor\n\n")
                elif len(original_shape) == 2: # Dense kernel
                    f_dense_k.write(header_comments)
                    f_dense_k.write("# Processing as Dense Kernel (row-major flattened)\n")
                    flattened_data = weights_data.flatten() 
                    for val_int8 in flattened_data:
                        # Ensure val_int8 is treated as a Python int for formatting
                        python_int_val = int(val_int8)
                        hex_val = format(python_int_val & 0xFF, '02x')
                        f_dense_k.write(hex_val + "\n")
                        total_dense_kernel_bytes += 1
                    f_dense_k.write("# End Tensor\n\n")
                else: # Other int8 tensors (if any) - unlikely for weights
                    print(f"INFO: Generic int8 tensor '{tensor_name}' (shape {original_shape}) not categorized as Conv or Dense kernel. Writing to conv_kernels file by default.")
                    f_conv_k.write(header_comments)
                    f_conv_k.write("# Processing as Generic int8 Tensor (row-major flattened)\n")
                    flattened_data = weights_data.flatten()
                    for val_int8 in flattened_data:
                        python_int_val = int(val_int8)
                        hex_val = format(python_int_val & 0xFF, '02x')
                        f_conv_k.write(hex_val + "\n")
                        total_conv_kernel_bytes += 1
                    f_conv_k.write("# End Tensor\n\n")

            elif data_type == np.int32: # Assumed to be Biases
                f_biases.write(header_comments)
                flattened_data = weights_data.flatten()
                for val_int32 in flattened_data:
                    try:
                        # Ensure val_int32 is a Python int before calling .to_bytes()
                        python_int_val = int(val_int32) # Handles np.int32 and other numeric types
                        
                        # Check standard int32 range for to_bytes, though Python's int handles arbitrary size
                        if python_int_val < -2147483648 or python_int_val > 2147483647:
                             print(f"Warning: int32 value {python_int_val} for tensor {tensor_name} is out of standard range for 4-byte signed conversion. Python's int.to_bytes will handle it.")
                        
                        byte_values = python_int_val.to_bytes(4, byteorder='little', signed=True)
                        for byte_val in byte_values:
                            hex_val = format(byte_val & 0xFF, '02x') # byte_val is already an int (0-255)
                            f_biases.write(hex_val + "\n")
                            total_bias_bytes += 1
                    except OverflowError as oe: # This might occur if python_int_val is too large for 4 signed bytes
                        print(f"OverflowError converting value {python_int_val} (from tensor {tensor_name}) to 4 bytes: {oe}. Skipping this value for bias file.")
                    except Exception as e_bytes:
                        print(f"Error converting value {python_int_val} (from tensor {tensor_name}) to bytes: {e_bytes}. Skipping this value for bias file.")
                f_biases.write("# End Tensor\n\n")
            else:
                print(f"WARNING: Skipping tensor {tensor_name} for hex output due to unhandled dtype: {data_type}")

    print(f"\nExtracted TFLite Conv2D kernel weights written to {output_conv_kernels_hex_file}")
    print(f"Total individual hex values (bytes) for Conv2D kernels: {total_conv_kernel_bytes}")
    print(f"\nExtracted TFLite Dense kernel weights written to {output_dense_kernels_hex_file}")
    print(f"Total individual hex values (bytes) for Dense kernels: {total_dense_kernel_bytes}")
    print(f"\nExtracted TFLite bias weights written to {output_biases_hex_file}")
    print(f"Total individual hex values (bytes) for biases: {total_bias_bytes}")
else:
    print("\nSkipping Step 6: No TFLite weights were successfully extracted in Step 3.")

# --- IMPORTANT CONSIDERATIONS FOR YOUR HARDWARE ---
# (Comments on Weight Order, Bias Handling, Scales, TFLite Converter remain the same)
# - Your hardware will need to know how to map the data from these three files to the correct layers.
# - The order of 4x4 planes in `tflite_conv_kernel_weights.hex` is:
#   Iterate output_channel, then iterate input_channel. Within each, the 4x4 is column-major.
# - Dense kernels in `tflite_dense_kernel_weights.hex` are row-major.
# - If your bias ROM expects int8 values, you'll need to further quantize the int32 biases.
