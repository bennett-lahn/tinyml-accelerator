`include "sys_types.svh"

// ======================================================================================================
// TPU DATAPATH
// ======================================================================================================
// This module implements the complete Tensor Processing Unit (TPU) datapath for the TinyML accelerator.
// It orchestrates the entire neural network inference pipeline, from input processing through
// convolutional layers, dense layers, and final classification with softmax.
//
// FUNCTIONALITY:
// - Processes convolutional layers using systolic tensor array (STA) with requantization and max pooling
// - Handles fully connected layers with dedicated dense layer computation engine
// - Manages data flow between layers using dual tensor RAM banks (ping-pong)
// - Performs spatial data formatting and unified buffer operations
// - Implements flattening operations for transition between conv and dense layers
// - Provides final classification with softmax probability computation
//
// ARCHITECTURE:
// - STA Controller: Core computation engine for convolutional operations
// - Unified Buffer: Spatial data extraction and sliding window operations
// - Tensor RAM: Dual-bank memory for layer-to-layer data storage
// - Weight/Bias ROMs: Storage for model parameters
// - Dense Layer Engine: Specialized computation for fully connected layers
// - Softmax Unit: Final classification with probability computation
//
// DATA FLOW:
// 1. Input data → Unified Buffer → Spatial Formatter → STA Controller
// 2. STA outputs → Requantization → Max Pooling → Tensor RAM
// 3. Tensor RAM → Unified Buffer → Next layer (conv layers)
// 4. Final conv layer → Flatten → Dense Layer Engine → Softmax
// 5. Softmax → Final classification probabilities
//
// LAYER TYPES:
// - Convolutional layers (0-3): Use STA with spatial processing
// - Dense layers (4-5): Use dedicated dense computation engine
// - Bypass modes: Skip maxpool/relu for specific layers
//
// MEMORY MANAGEMENT:
// - Dual tensor RAM banks for ping-pong operation
// - Layer-based memory switching (layer_idx[0] determines active bank)
// - Unified buffer for spatial data extraction
// - Dedicated ROMs for weights and biases
//
// PARAMETERS:
// - IMG_W/H: Input image dimensions
// - MAX_N: Maximum matrix dimension
// - NUM_LAYERS: Total number of layers in model
// - ROM parameters: Memory depths and widths for weights/biases
// ======================================================================================================

module TPU_Datapath #(
    parameter IMG_W                = 32                     // Default MAX_N from sta_controller
    ,parameter IMG_H               = 32
    ,parameter MAX_N               = 64                     // Default MAX_N from sta_controller
    ,parameter MAX_NUM_CH          = 64
    ,parameter STA_CTRL_N_BITS     = $clog2(MAX_N + 1)      // Should be 9 bits for MAX_N=512
    ,parameter SA_N                = 4                      // SA_N from sta_controller
    ,parameter SA_VECTOR_WIDTH     = 4                      // SA_VECTOR_WIDTH from sta_controller
    ,parameter TOTAL_PES_STA       = SA_N * SA_N            // Should be 16
    ,parameter NUM_LAYERS          = 6                      // Number of layers in the model
    ,parameter MAX_PADDING         = 3                      // Max padding for any side
    ,parameter int MAX_BYPASS_IDX  = 64                     // Max index value for bypass mode (fully connected layers)
    ,parameter int BYPASS_IDX_BITS = $clog2(MAX_BYPASS_IDX) // Bits to hold bypass index
    ,parameter int RAM_WORD_DEPTH  = 128                    // 128 words of 128 bits each to store first layer (largest layer)
    ,parameter int RAM_READ_WIDTH  = 128
    ,parameter int RAM_WRITE_WIDTH = 8
    ,parameter int ROM_WEIGHT_DEPTH = 2696
    ,parameter int ROM_WEIGHT_WIDTH = 128
    ,parameter int ROM_BIAS_DEPTH   = 240
    ,parameter int ROM_BIAS_WIDTH   = 32
)
(
    input logic clk
    ,input logic reset

    ,input logic [$clog2(NUM_LAYERS)-1:0] layer_idx // Current layer index
    ,input logic [$clog2(MAX_NUM_CH)-1:0] channel_idx // Current output channel index

    ,input logic read_bias
    ,input logic load_bias

    ,input logic reset_sta
    ,input logic reset_datapath
    //starts weights and activation COMPUTATION
    ,input logic start // starts activation and weight input to STA
    ,input logic done

    //unified buffer configuration
    ,input logic [$clog2(64+1)-1:0] img_width
    ,input logic [$clog2(64+1)-1:0] img_height
    ,input logic [$clog2(64+1)-1:0] num_channels_input
    ,input logic [$clog2(MAX_N+1)-1:0] num_columns_output
    ,input logic [$clog2(64+1)-1:0] num_channels_output

    ,input [$clog2(MAX_PADDING+1)-1:0] pad_top
    ,input [$clog2(MAX_PADDING+1)-1:0] pad_bottom
    ,input [$clog2(MAX_PADDING+1)-1:0] pad_left
    ,input [$clog2(MAX_PADDING+1)-1:0] pad_right

    //unified buffer control signals
    ,input logic start_block_extraction
    ,input logic next_channel_group
    ,input logic next_spatial_block

    ,input logic start_flatten
    ,input logic flatten_stage

    ,input logic read_logits
    ,input logic softmax_start

    //dense layer
    ,input logic start_dense_compute
    ,input logic input_valid_dense

    // Runtime configuration inputs
    ,input logic [$clog2(256+1)-1:0] input_size_dense    // Actual input size (up to 256)
    ,input logic [$clog2(64+1)-1:0] output_size_dense

    // These three signals do very similar things and may be redundant
    ,output logic all_cols_sent
    ,output logic all_channels_done
    ,output logic patches_valid

    ,output logic sta_idle
    ,output logic dense_compute_completed
    ,output logic flatten_complete
    ,output logic softmax_valid
    
    ,output logic [31:0] probabilities_o [9:0] // Index of the output value in the array
);
// ======================================================================================================
// LAYER CONTROLLER
// ======================================================================================================

// layer_controller layer_controller (
//     .clk(clk)
//     ,.reset(reset)

//     // Inputs to layer_controller
//     ,.start(start)
//     ,.stall(stall)
//     ,.sta_idle(sta_idle)
//     ,.done(done)
//     ,.layer_idx(layer_idx)
//     ,.chnnl_idx(channel_idx)
//     ,.reset_sta(reset_sta)

//     // Outputs from layer_controller
//     ,.mat_size(mat_size)
//     ,.load_bias(load_bias)
//     ,.start_compute(start_compute)
//     ,.controller_pos_row(controller_pos_row)
//     ,.controller_pos_col(controler_pos_col)
//     ,.num_filters(num_filters)
//     ,.num_input_channels(num_input_channels)
// );

// ======================================================================================================
// STA CONTROLLER
// ======================================================================================================
// The STA controller orchestrates the systolic tensor array computation pipeline, including
// matrix multiplication, requantization, and max pooling. It serves as the primary computation
// engine for convolutional layers (layers 0-3).

// Internal control and data signals for STA controller
logic                          stall;                         // Stall signal for flow control
int32_t                        bias_value;                   // Bias value loaded from ROM for all PEs

// STA controller output signals
logic                          idle;                         // High when STA complex is idle
logic                          array_out_valid;              // Valid signal for output data
int8_t                         array_val_out;                // 8-bit quantized output value
logic [$clog2(MAX_N)-1:0]      array_row_out;                // Spatial row coordinate
logic [$clog2(MAX_N)-1:0]      array_col_out;                // Spatial column coordinate

// Layer configuration parameters
logic [15:0]                   mat_size;                     // Matrix size for current layer
logic                          start_compute;                // Start computation signal
logic [$clog2(MAX_N)-1:0]      array_index_out;              // Index for bypass mode

// Current layer filter dimensions
logic [$clog2(MAX_NUM_CH+1)-1:0] num_filters;                // Number of output channels
logic [$clog2(MAX_NUM_CH+1)-1:0] num_input_channels;         // Number of input channels

// Bypass mode signals for dense layer integration
logic                            bypass_valid;               // Valid signal for bypass data
logic [BYPASS_IDX_BITS-1:0]      bypass_index;               // Index for bypass value
int32_t                          bypass_value;               // Bypass data from dense layer

// Controller position signals (derived from unified buffer)
logic [$clog2(MAX_N)-1:0]        controller_pos_row;         // Base row position for current block
logic [$clog2(MAX_N)-1:0]        controller_pos_col;         // Base column position for current block

// Block address signals from unified buffer
logic [$clog2(MAX_N)-1:0] block_start_col_addr;              // Write address for tensor RAM A
logic [$clog2(MAX_N)-1:0] block_start_row_addr;              // Write address for tensor RAM B

// Connect controller position to block addresses from unified buffer
assign controller_pos_row = block_start_row_addr;            // Use block start row for controller
assign controller_pos_col = block_start_col_addr;            // Use block start column for controller

// Weight matrix inputs for STA (loaded by weight_loader)
int8_t B0 [SA_VECTOR_WIDTH];                                  // Weight vector for column 0
int8_t B1 [SA_VECTOR_WIDTH];                                  // Weight vector for column 1
int8_t B2 [SA_VECTOR_WIDTH];                                  // Weight vector for column 2
int8_t B3 [SA_VECTOR_WIDTH];                                  // Weight vector for column 3

sta_controller sta_controller (
    .clk(clk)
    ,.reset(reset)
    ,.reset_sta(reset_sta)
    ,.stall('0)

    ,.bypass_maxpool(bypass_maxpool)
    ,.bypass_relu(bypass_relu)
    ,.bypass_valid(bypass_valid)
    ,.bypass_value(bypass_value)
    ,.bypass_index(bypass_index)

    ,.controller_pos_row(controller_pos_row)
    ,.controller_pos_col(controller_pos_col)
    ,.done(done)
    ,.layer_idx(layer_idx)
    ,.A0(formatted_A0)
    ,.A1(formatted_A1)
    ,.A2(formatted_A2)
    ,.A3(formatted_A3)
    ,.B0(B0)
    ,.B1(B1)
    ,.B2(B2)
    ,.B3(B3)
    ,.load_bias(load_bias)
    ,.bias_value(bias_value)

    ,.idle(idle)
    ,.sta_idle(sta_idle)
    ,.array_out_valid(array_out_valid)
    ,.array_val_out(array_val_out)
    ,.array_row_out(array_row_out)
    ,.array_col_out(array_col_out)
    ,.array_index_out(array_index_out)
);

//======================================================================================================
// BIAS ROM - Layer Bias Storage
//======================================================================================================
// The bias ROM stores bias values for each layer and channel in the neural network.
// It provides the bias values that are added to the accumulator results in the STA
// during the computation process.

bias_rom #(
    .WIDTH(ROM_BIAS_WIDTH)                                   // 32-bit bias values
    ,.DEPTH(ROM_BIAS_DEPTH)                                  // 240 bias entries
    ,.MAX_NUM_CH(MAX_NUM_CH)                                 // Maximum number of channels
    ,.INIT_FILE("../fakemodel/tflite_bias_weights.hex")      // Bias weights from TFLite model
) BIAS_ROM (
    .clk(clk)
    ,.read_enable(read_bias)
    ,.layer_idx(layer_idx)
    ,.channel_idx(channel_idx)
    ,.bias_out(bias_value)
);

//======================================================================================================
// TENSOR RAM - Dual-Bank Layer Memory System
//======================================================================================================
// The tensor RAM implements a dual-bank memory system for layer-to-layer data storage.
// RAM A and RAM B operate in ping-pong fashion - while one bank is being written to
// by the current layer, the other bank is being read from for the next layer.
// This prevents read/write conflicts.

// Tensor RAM A - First memory bank
// Data outputs from RAM A (4-wide read interface)
int32_t tensor_ram_A_dout0;                                   // Output data word 0
int32_t tensor_ram_A_dout1;                                   // Output data word 1
int32_t tensor_ram_A_dout2;                                   // Output data word 2
int32_t tensor_ram_A_dout3;                                   // Output data word 3
int8_t tensor_ram_A_din;                                      // Input data (8-bit write)
logic tensor_ram_A_data_valid;                                // Data valid signal
logic ram_A_we;                                               // Write enable for RAM A
logic ram_A_re;                                               // Read enable for RAM A
logic [($clog2(RAM_WORD_DEPTH*(RAM_READ_WIDTH/RAM_WRITE_WIDTH)))-1:0] ram_A_addr_r; // Read address

// Instantiate tensor RAM A with initial data (first layer input)
tensor_ram #(
    .INIT_FILE("../fakemodel/test_vector_dog.hex")            // Initial input image data
) RAM_A (
    .clk(clk)
    ,.reset(reset)
    ,.write_en(ram_A_we)
    ,.read_en(ram_A_re)
    ,.write_row(array_row_out)
    ,.write_col(array_col_out)
    ,.write_channel(channel_idx)
    ,.num_cols(num_columns_output)
    ,.num_channels(num_channels_input)
    ,.data_in(tensor_ram_A_din)
    ,.read_addr(ram_A_addr_r)
    ,.ram_dout0(tensor_ram_A_dout0)
    ,.ram_dout1(tensor_ram_A_dout1)
    ,.ram_dout2(tensor_ram_A_dout2)
    ,.ram_dout3(tensor_ram_A_dout3)
    ,.data_valid(tensor_ram_A_data_valid)
);

// Tensor RAM B - Second memory bank
// Data outputs from RAM B (4-wide read interface)
int32_t tensor_ram_B_dout0;                                   // Output data word 0
int32_t tensor_ram_B_dout1;                                   // Output data word 1
int32_t tensor_ram_B_dout2;                                   // Output data word 2
int32_t tensor_ram_B_dout3;                                   // Output data word 3
int8_t tensor_ram_B_din;                                      // Input data (8-bit write)
logic tensor_ram_B_data_valid;                                // Data valid signal
logic ram_B_we;                                               // Write enable for RAM B
logic ram_B_re;                                               // Read enable for RAM B
logic [($clog2(RAM_WORD_DEPTH*(RAM_READ_WIDTH/RAM_WRITE_WIDTH)))-1:0] ram_B_addr_r; // Read address

// Instantiate tensor RAM B (empty initialization for intermediate layers)
tensor_ram #(
    .INIT_FILE("")                                             // No initial data (intermediate storage)
) RAM_B (
    .clk(clk)
    ,.reset(reset)
    ,.write_en(ram_B_we)
    ,.read_en(ram_B_re)
    ,.write_row(array_row_out)
    ,.write_col(array_col_out)
    ,.write_channel(channel_idx)
    ,.num_cols(num_columns_output)
    ,.num_channels(num_channels_output)
    ,.data_in(tensor_ram_B_din)
    ,.read_addr(ram_B_addr_r)
    ,.ram_dout0(tensor_ram_B_dout0)
    ,.ram_dout1(tensor_ram_B_dout1)
    ,.ram_dout2(tensor_ram_B_dout2)
    ,.ram_dout3(tensor_ram_B_dout3)
    ,.data_valid(tensor_ram_B_data_valid)
);

//======================================================================================================
// TENSOR RAM CONTROL LOGIC
//======================================================================================================
// This section implements the control logic for the dual-bank tensor RAM system.
// The layer_idx[0] bit determines which RAM bank is active for writing/reading.
// Even layers (layer_idx[0]=0) use RAM A for writing, RAM B for reading.
// Odd layers (layer_idx[0]=1) use RAM B for writing, RAM A for reading.

// Write enable control based on layer index
assign ram_A_we = layer_idx[0] & array_out_valid;             // Write to RAM A on odd layers
assign ram_B_we = ~layer_idx[0] & array_out_valid;            // Write to RAM B on even layers

// Read enable control with flatten stage support
assign ram_A_re = flatten_stage ? request_chunk :              // Flatten stage: read for flattening
                  ~layer_idx[0] ? buffer_ram_read_enable :     // Even layers: read from RAM A
                  1'b0;                                        // Odd layers: no read from RAM A
assign ram_B_re = layer_idx[0] ? buffer_ram_read_enable :      // Odd layers: read from RAM B
                  1'b0;                                        // Even layers: no read from RAM B

// Read address selection
assign ram_A_addr_r = flatten_stage ? chunk_addr :             // Flatten stage: use chunk address
                      ~layer_idx[0] ? buffer_ram_addr :        // Even layers: use buffer address
                      0;                                        // Odd layers: no read
assign ram_B_addr_r = layer_idx[0] ? buffer_ram_addr :         // Odd layers: use buffer address
                      0;                                        // Even layers: no read

// Combined data valid signal from both RAM banks
logic ram_data_valid;
assign ram_data_valid = tensor_ram_A_data_valid | tensor_ram_B_data_valid;

// Data routing logic - select appropriate RAM bank based on layer index
always_comb begin
    // Route read data to unified buffer based on active RAM bank
    buffer_ram_in0 = layer_idx[0] ? tensor_ram_B_dout0 : tensor_ram_A_dout0; // Select RAM B for odd layers
    buffer_ram_in1 = layer_idx[0] ? tensor_ram_B_dout1 : tensor_ram_A_dout1; // Select RAM B for odd layers
    buffer_ram_in2 = layer_idx[0] ? tensor_ram_B_dout2 : tensor_ram_A_dout2; // Select RAM B for odd layers
    buffer_ram_in3 = layer_idx[0] ? tensor_ram_B_dout3 : tensor_ram_A_dout3; // Select RAM B for odd layers

    // Route write data to appropriate RAM bank based on layer index
    if(layer_idx[0] == 0) begin
        // Even layers: write to RAM B, no write to RAM A
        tensor_ram_B_din = (array_out_valid) ? array_val_out : 'd0;
        tensor_ram_A_din = 'd0;
    end else begin
        // Odd layers: write to RAM A, no write to RAM B
        tensor_ram_A_din = (array_out_valid) ? array_val_out : 'd0;
        tensor_ram_B_din = 'd0;
    end
end

//======================================================================================================
// UNIFIED BUFFER - Spatial Data Extraction and Processing
//======================================================================================================
// The unified buffer extracts spatial blocks of data from tensor RAM and provides
// sliding window functionality for convolutional operations. It generates 7x7 patches
// that are used by the spatial data formatter to create inputs for the STA.

// Unified buffer control and addressing signals
logic buffer_ram_read_enable;                                // Read enable for tensor RAM
logic [$clog2(64*64*64/4)-1:0] buffer_ram_addr;              // Read address for tensor RAM

// Unified buffer input data from tensor RAM (4-wide interface)
logic [31:0] buffer_ram_in0;                                  // Input data word 0
logic [31:0] buffer_ram_in1;                                  // Input data word 1
logic [31:0] buffer_ram_in2;                                  // Input data word 2
logic [31:0] buffer_ram_in3;                                  // Input data word 3

// Unified buffer output - 7x7 spatial patch positions
// Each patch_peXX_out represents one position in the 7x7 sliding window
logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out, patch_pe04_out, patch_pe05_out, patch_pe06_out;
logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out, patch_pe14_out, patch_pe15_out, patch_pe16_out;
logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out, patch_pe24_out, patch_pe25_out, patch_pe26_out;
logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out, patch_pe34_out, patch_pe35_out, patch_pe36_out;
logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out, patch_pe44_out, patch_pe45_out, patch_pe46_out;
logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out, patch_pe54_out, patch_pe55_out, patch_pe56_out;
logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out, patch_pe64_out, patch_pe65_out, patch_pe66_out;

// Instantiate unified buffer harness for spatial block extraction
unified_buffer_harness #(
    .MAX_IMG_W(IMG_W)                                         // Maximum image width
    ,.MAX_IMG_H(IMG_H)                                        // Maximum image height
    ,.MAX_CHANNELS(MAX_NUM_CH)                                // Maximum number of channels
    ,.MAX_PADDING(MAX_PADDING)                                // Maximum padding size
) UNIFIED_BUFFER_HARNESS (
    .clk(clk)
    ,.reset(reset | reset_datapath)
    ,.current_layer_idx(layer_idx)
    ,.start_extraction(start_block_extraction)
    ,.next_channel_group(next_channel_group)
    ,.next_spatial_block(next_spatial_block)
    ,.ram_re(buffer_ram_read_enable)
    ,.ram_addr(buffer_ram_addr)
    ,.ram_dout0(buffer_ram_in0)
    ,.ram_dout1(buffer_ram_in1)
    ,.ram_dout2(buffer_ram_in2)
    ,.ram_dout3(buffer_ram_in3)
    ,.ram_data_valid(ram_data_valid)
    ,.all_channels_done(all_channels_done)
    ,.block_start_col_addr(block_start_col_addr)
    ,.block_start_row_addr(block_start_row_addr)
     // All 7x7 position outputs
    ,.patch_pe00_out(patch_pe00_out), .patch_pe01_out(patch_pe01_out), .patch_pe02_out(patch_pe02_out), .patch_pe03_out(patch_pe03_out), .patch_pe04_out(patch_pe04_out), .patch_pe05_out(patch_pe05_out), .patch_pe06_out(patch_pe06_out)
    ,.patch_pe10_out(patch_pe10_out), .patch_pe11_out(patch_pe11_out), .patch_pe12_out(patch_pe12_out), .patch_pe13_out(patch_pe13_out), .patch_pe14_out(patch_pe14_out), .patch_pe15_out(patch_pe15_out), .patch_pe16_out(patch_pe16_out)
    ,.patch_pe20_out(patch_pe20_out), .patch_pe21_out(patch_pe21_out), .patch_pe22_out(patch_pe22_out), .patch_pe23_out(patch_pe23_out), .patch_pe24_out(patch_pe24_out), .patch_pe25_out(patch_pe25_out), .patch_pe26_out(patch_pe26_out)
    ,.patch_pe30_out(patch_pe30_out), .patch_pe31_out(patch_pe31_out), .patch_pe32_out(patch_pe32_out), .patch_pe33_out(patch_pe33_out), .patch_pe34_out(patch_pe34_out), .patch_pe35_out(patch_pe35_out), .patch_pe36_out(patch_pe36_out)
    ,.patch_pe40_out(patch_pe40_out), .patch_pe41_out(patch_pe41_out), .patch_pe42_out(patch_pe42_out), .patch_pe43_out(patch_pe43_out), .patch_pe44_out(patch_pe44_out), .patch_pe45_out(patch_pe45_out), .patch_pe46_out(patch_pe46_out)
    ,.patch_pe50_out(patch_pe50_out), .patch_pe51_out(patch_pe51_out), .patch_pe52_out(patch_pe52_out), .patch_pe53_out(patch_pe53_out), .patch_pe54_out(patch_pe54_out), .patch_pe55_out(patch_pe55_out), .patch_pe56_out(patch_pe56_out)
    ,.patch_pe60_out(patch_pe60_out), .patch_pe61_out(patch_pe61_out), .patch_pe62_out(patch_pe62_out), .patch_pe63_out(patch_pe63_out), .patch_pe64_out(patch_pe64_out), .patch_pe65_out(patch_pe65_out), .patch_pe66_out(patch_pe66_out)
    ,.patches_valid(patches_valid)
);

//======================================================================================================
// SPATIAL DATA FORMATTER - STA Input Preparation
//======================================================================================================
// The spatial data formatter converts 7x7 spatial patches from the unified buffer
// into the 4x4 format required by the systolic tensor array. It extracts the
// appropriate 4x4 region and formats it for STA processing.

// Formatted activation inputs for STA (4x4 matrix format)
int8_t formatted_A0 [0:3];                                    // Row 0, 4 channels of current column position
int8_t formatted_A1 [0:3];                                    // Row 1, 4 channels of current column position  
int8_t formatted_A2 [0:3];                                    // Row 2, 4 channels of current column position
int8_t formatted_A3 [0:3];                                    // Row 3, 4 channels of current column position

// Instantiate spatial data formatter
spatial_data_formatter SPATIAL_FORMATTER (
    .clk(clk)
    ,.reset(reset | reset_datapath)
    ,.start_formatting(start & patches_valid)
    ,.patches_valid(patches_valid)
    // All 7x7 position inputs
    ,.patch_pe00_in(patch_pe00_out), .patch_pe01_in(patch_pe01_out), .patch_pe02_in(patch_pe02_out), .patch_pe03_in(patch_pe03_out), .patch_pe04_in(patch_pe04_out), .patch_pe05_in(patch_pe05_out), .patch_pe06_in(patch_pe06_out)
    ,.patch_pe10_in(patch_pe10_out), .patch_pe11_in(patch_pe11_out), .patch_pe12_in(patch_pe12_out), .patch_pe13_in(patch_pe13_out), .patch_pe14_in(patch_pe14_out), .patch_pe15_in(patch_pe15_out), .patch_pe16_in(patch_pe16_out)
    ,.patch_pe20_in(patch_pe20_out), .patch_pe21_in(patch_pe21_out), .patch_pe22_in(patch_pe22_out), .patch_pe23_in(patch_pe23_out), .patch_pe24_in(patch_pe24_out), .patch_pe25_in(patch_pe25_out), .patch_pe26_in(patch_pe26_out)
    ,.patch_pe30_in(patch_pe30_out), .patch_pe31_in(patch_pe31_out), .patch_pe32_in(patch_pe32_out), .patch_pe33_in(patch_pe33_out), .patch_pe34_in(patch_pe34_out), .patch_pe35_in(patch_pe35_out), .patch_pe36_in(patch_pe36_out)
    ,.patch_pe40_in(patch_pe40_out), .patch_pe41_in(patch_pe41_out), .patch_pe42_in(patch_pe42_out), .patch_pe43_in(patch_pe43_out), .patch_pe44_in(patch_pe44_out), .patch_pe45_in(patch_pe45_out), .patch_pe46_in(patch_pe46_out)
    ,.patch_pe50_in(patch_pe50_out), .patch_pe51_in(patch_pe51_out), .patch_pe52_in(patch_pe52_out), .patch_pe53_in(patch_pe53_out), .patch_pe54_in(patch_pe54_out), .patch_pe55_in(patch_pe55_out), .patch_pe56_in(patch_pe56_out)
    ,.patch_pe60_in(patch_pe60_out), .patch_pe61_in(patch_pe61_out), .patch_pe62_in(patch_pe62_out), .patch_pe63_in(patch_pe63_out), .patch_pe64_in(patch_pe64_out), .patch_pe65_in(patch_pe65_out), .patch_pe66_in(patch_pe66_out)
    ,.formatted_A0(formatted_A0)
    ,.formatted_A1(formatted_A1)
    ,.formatted_A2(formatted_A2)
    ,.formatted_A3(formatted_A3)
    ,.all_cols_sent(all_cols_sent)
);

//======================================================================================================
// WEIGHT DATAPATH - Convolutional Weight Loading System
//======================================================================================================
// The weight datapath manages the loading of convolutional weights from ROM into
// the systolic tensor array. It coordinates with the spatial data formatter to
// provide the correct weights for each computation cycle.

// Weight ROM interface signals
logic weight_rom_read_enable;                                 // Read enable for weight ROM
logic [$clog2(ROM_WEIGHT_WIDTH)-1:0] weight_rom_addr;        // Read address for weight ROM
int32_t weight_rom_dout0;                                     // Weight data output 0
int32_t weight_rom_dout1;                                     // Weight data output 1
int32_t weight_rom_dout2;                                     // Weight data output 2
int32_t weight_rom_dout3;                                     // Weight data output 3

// Weight loader control signals
logic weight_load_idle;                                       // Weight loader idle signal
logic weight_load_complete;                                   // Weight loading complete signal

// Instantiate weight loader for STA weight management
weight_loader # (
    .MAX_NUM_CH(MAX_NUM_CH)                                   // Maximum number of channels
    ,.SA_N(SA_N)                                              // Systolic array dimension
    ,.VECTOR_WIDTH(SA_VECTOR_WIDTH)                           // Vector width for weights
    ,.ROM_DEPTH(ROM_WEIGHT_DEPTH)                             // Weight ROM depth
) weight_loader (
    .clk(clk)
    ,.reset(reset | reset_datapath | reset_sta)
    ,.start(start & patches_valid)
    ,.stall(stall)
    ,.next_channel_group(next_channel_group)
    
    ,.output_channel_idx(channel_idx)
    ,.current_layer_idx(layer_idx)

    ,.weight_rom_read_enable(weight_rom_read_enable)
    ,.weight_rom_addr(weight_rom_addr)
    ,.weight_rom_data0(weight_rom_dout0)
    ,.weight_rom_data1(weight_rom_dout1)
    ,.weight_rom_data2(weight_rom_dout2)
    ,.weight_rom_data3(weight_rom_dout3)

    ,.B0(B0)
    ,.B1(B1)
    ,.B2(B2)
    ,.B3(B3)

    ,.idle(weight_load_idle)
    ,.weight_load_complete(weight_load_complete)
);

    // ROM for weights, assuming 16 8-bit values per read
    weight_rom  #(
        .WIDTH(ROM_WEIGHT_WIDTH) // 4 8-bit pixels per read/write
        ,.DEPTH(ROM_WEIGHT_DEPTH) // Assuming IMG_W and IMG_H are defined in the module scope
        ,.INIT_FILE("../fakemodel/tflite_conv_kernel_weights.hex") // No initialization file specified
    ) weight_rom (
        .clk(clk)
        ,.read_enable(weight_rom_read_enable)
        ,.addr(weight_rom_addr)
        ,.data0(weight_rom_dout0)
        ,.data1(weight_rom_dout1)
        ,.data2(weight_rom_dout2)
        ,.data3(weight_rom_dout3)
    );

//======================================================================================================
// FLATTEN - Convolutional to Dense Layer Transition
//======================================================================================================
// The flatten layer converts the 2D spatial output from the final convolutional layer
// into a 1D vector suitable for dense layer processing. It reads data from tensor RAM
// in chunks and outputs individual elements for dense layer computation.

// Flatten layer control and data signals
logic chunk_valid;                                           // Indicates if current chunk is valid
logic [127:0] input_chunk;                                   // 128-bit input chunk (16 int8_t values)
logic request_chunk;                                         // Request next 128-bit chunk
logic [$clog2(64)-1:0] chunk_addr;                           // Current chunk address (0-63)
logic output_read_enable;                                    // Enable reading output data
logic [31:0] input_data [0:64-1];                            // Input data from previous layer
logic [$clog2(2)-1:0] input_row;                             // Input row coordinate
logic [$clog2(2)-1:0] input_col;                             // Input column coordinate

// Flatten layer output signals
int8_t output_data_flatten;                                  // Output data (one element at a time)
logic [$clog2(256)-1:0] output_addr_flatten;                 // Current output address
logic output_valid;                                          // Indicates output data is valid
logic all_outputs_sent;                                      // Indicates all data has been output

flatten_layer #(
    .INPUT_HEIGHT(2)                                         // Input height (2x2 spatial)
    ,.INPUT_WIDTH(2)                                         // Input width (2x2 spatial)
    ,.INPUT_CHANNELS(MAX_NUM_CH)                             // Number of input channels
    ,.CHUNK_SIZE(16)                                         // 128 bits = 16 int8_t values per chunk
    ,.TOTAL_CHUNKS(16)                                       // Total chunks based on input dimensions
    ,.OUTPUT_SIZE(IMG_H * IMG_W * MAX_NUM_CH)                // Total output size
) flatten_layer (
    .clk(clk)
    ,.reset(reset)

    // Control signals
    ,.start_flatten(start_flatten)
    ,.output_read_enable(output_read_enable)

    // Input data (128-bit chunk)
    ,.input_chunk(input_chunk)
    ,.chunk_valid(chunk_valid)

    // Output data (one pixel at a time)
    ,.output_data(output_data_flatten)
    ,.output_addr(output_addr_flatten)
    ,.output_valid(output_valid)
    ,.flatten_complete(flatten_complete)

    // Memory request signals
    ,.request_chunk(request_chunk)
    ,.chunk_addr(chunk_addr)
);

assign input_chunk = flatten_stage ? {tensor_ram_A_dout0, tensor_ram_A_dout1, tensor_ram_A_dout2, tensor_ram_A_dout3} : 0;
assign chunk_valid = flatten_stage;

//======================================================================================================
// DENSE fully connected ram
//======================================================================================================

// The dense layer compute engine performs matrix-vector multiplication for fully
// connected layers. It interfaces with the dense FC RAM, weight ROM, and bias ROM
// to compute dense layer outputs one at a time.

    logic dense_fc_write_enable;
    logic [($clog2(256))-1:0] dense_fc_read_addr;
    logic [($clog2(256))-1:0] dense_fc_write_addr;
    logic [7:0] dense_fc_data_in;
    logic [7:0] dense_fc_data_out;
    logic dense_fc_read_enable;

    assign dense_fc_data_in = (flatten_stage) ? output_data_flatten : (layer_idx > 'd3) ? array_val_out : '0; // Input data for dense fully connected layer
    assign dense_fc_write_enable = (flatten_stage & output_valid) ? 1'b1 : (layer_idx > 'd3 && array_out_valid); // Write enable for dense fully connected layer
    // assign dense_fc_read_enable = read_logits | (flatten_stage & output_valid); // Read enable for dense fully connected layer
    assign dense_fc_write_addr = (flatten_stage) ? output_addr_flatten : (layer_idx > 'd3) ? array_index_out : '0; // Write address for dense fully connected layer

    dense_fc_ram #(
        .DEPTH(256) // 256 words of 8 bits each
        ,.WIDTH(8) // 8 bits per word
        ,.INIT_FILE("") // Initialization file for weights
    ) dense_fc_ram (
        .clk(clk)
        ,.reset(reset)
        ,.write_enable(dense_fc_write_enable)
        ,.read_addr((read_logits) ? logit_load_count : dense_fc_read_addr)
        ,.write_addr(dense_fc_write_addr)
        ,.read_enable(dense_fc_read_enable | read_logits)
        ,.data_in(dense_fc_data_in)
        ,.data_out(dense_fc_data_out)
    );


//======================================================================================================
// FC_bias_rom

    logic fc_bias_rom_read_enable;
    logic fc_layer_select;
    logic [$clog2(72)-1:0] fc_bias_rom_addr; // Address for bias ROM
    logic [31:0] fc_bias_rom_dout; // Output data from the bias ROM

    fc_bias_rom #(
        .WIDTH(32) // 32 bits per word
        ,.DEPTH(74) // 74 words for the fully connected layer
        ,.INIT_FILE("../fakemodel/tflite_fc_biases.hex") // Initialization file for bias weights
        ,.FC1_SIZE(64) // Size of the first fully connected layer
        ,.FC2_SIZE(10) // Size of the second fully connected layer
    ) fc_bias_rom (
        .clk(clk)
        ,.read_enable(fc_bias_rom_read_enable)
        ,.fc_layer_select(fc_layer_select)
        ,.addr(fc_bias_rom_addr)
        ,.bias_out(fc_bias_rom_dout)
    );


//======================================================================================================
// DENSE WEIGHT ROM
//======================================================================================================

    logic dense_weight_rom_read_enable;
    logic [$clog2(256*64)-1:0] dense_weight_rom_addr; // Address for weight ROM
    logic [7:0] dense_weight_rom_dout; // Output data from the weight ROM

    dense_weight_rom #(
        .WIDTH(8) // 8 bits per word
        ,.DEPTH(17024) // 256 input channels, 64 output channels, then 64 input channels, 10 output channels
        ,.INIT_FILE("../fakemodel/tflite_dense_kernel_weights.hex") // Initialization file for weights
    ) dense_weight_rom (
        .clk(clk)
        ,.reset(reset)
        ,.read_enable(dense_weight_rom_read_enable)
        ,.addr(dense_weight_rom_addr)
        ,.weight_out(dense_weight_rom_dout)
    );

//======================================================================================================
    
    // Tensor RAM interface (for input vectors)
    logic [$clog2(256)-1:0] tensor_ram_addr;
    logic tensor_ram_re;
    logic [7:0] tensor_ram_dout;
    
    // Weight ROM interface (for weight matrix)
    logic [$clog2(256*64)-1:0] weight_rom_addr_dense;
    logic weight_rom_re;
    logic [7:0] weight_rom_dout;
    
    // Bias ROM interface (for bias vectors)
    logic [$clog2(64)-1:0] bias_rom_addr;
    logic bias_rom_re;
    logic [31:0] bias_rom_dout;
    
    // Single output (one result at a time)
    logic [31:0] output_data;
    logic [$clog2(64)-1:0] output_channel;  // Which output channel this result is for
    logic [$clog2(64)-1:0] output_addr; // Current input size for this layer
    logic output_ready;                    // Indicates output_data is valid for current channel

dense_layer_compute dense_compute (
    .clk(clk)
    ,.reset(reset)
    // Control signals
    ,.start_compute(start_dense_compute)
    ,.input_valid(input_valid_dense)
    // Runtime configuration inputs
    ,.input_size(input_size_dense)
    ,.output_size(output_size_dense)
    // Tensor RAM interface (for input vectors)
    ,.tensor_ram_addr(tensor_ram_addr)
    ,.tensor_ram_re(tensor_ram_re)
    ,.tensor_ram_dout(tensor_ram_dout)
    // Weight ROM interface (for weight matrix)
    ,.weight_rom_addr(weight_rom_addr_dense)
    ,.weight_rom_re(weight_rom_re)
    ,.weight_rom_dout(weight_rom_dout)
    // Bias ROM interface (for bias vectors)
    ,.bias_rom_addr(bias_rom_addr)
    ,.bias_rom_re(bias_rom_re)
    ,.bias_rom_dout(bias_rom_dout)
    // Single output (one result at a time)
    ,.output_data(output_data) 
    ,.output_channel(output_channel)  // Which output channel this result is for
    ,.output_addr(output_addr)
    ,.output_ready(output_ready)                     // Indicates output_data is valid for current channel
    ,.computation_complete(dense_compute_completed)     // All outputs computed
);

//======================================================================================================
// BYPASS LOGIC - Layer-Specific Control
//======================================================================================================
// Bypass logic controls which operations are skipped for specific layers.
// This enables efficient processing of different layer types.

// Bypass control signals
logic bypass_maxpool;                                         // Skip max pooling for dense layers
logic bypass_relu;                                            // Skip ReLU for final classification layer

// Layer-specific bypass assignments
assign bypass_maxpool = (layer_idx > 'd3);                    // Skip maxpool for layers 4-5 (dense)
assign bypass_relu = (layer_idx == 'd5);                      // Skip ReLU for layer 5 (final dense)

// Connect dense layer outputs to STA bypass interface
assign fc_bias_rom_read_enable = bias_rom_re;                 // Connect bias ROM read enable
assign fc_bias_rom_addr = bias_rom_addr;                      // Connect bias ROM address
assign bias_rom_dout = fc_bias_rom_dout;                      // Connect bias ROM data
assign dense_weight_rom_read_enable = weight_rom_re;          // Connect weight ROM read enable
assign dense_weight_rom_addr = weight_rom_addr_dense;         // Connect weight ROM address
assign weight_rom_dout = dense_weight_rom_dout;               // Connect weight ROM data
assign dense_fc_read_enable = tensor_ram_re;                  // Connect dense FC RAM read enable
assign dense_fc_read_addr = tensor_ram_addr;                  // Connect dense FC RAM address
assign tensor_ram_dout = dense_fc_data_out;                   // Connect dense FC RAM data

// Connect dense computation outputs to STA bypass
assign bypass_valid = output_ready;                           // Valid signal for bypass data
assign bypass_value = output_data;                            // Bypass data value
assign bypass_index = output_addr;                            // Bypass index (output address)

//======================================================================================================
// SOFTMAX - Final Classification Layer
//======================================================================================================
// The softmax unit performs the final classification by converting logits from the
// last dense layer into probability distributions across the 10 output classes.

// Softmax control and data signals
localparam int NUM_LOGITS = 10;                               // Number of classification classes
logic [$clog2(NUM_LOGITS)-1:0] logit_load_count;              // Counter for loading logits
int8_t logit_buffer [NUM_LOGITS];                             // Buffer for storing logits
logic wait_for_read;                                          // Wait signal for initial load delay

// Logit loading logic - loads dense layer outputs into buffer
always_ff @(posedge clk) begin
    if (reset) begin
        logit_load_count <= '0;
        wait_for_read <= 1'b0;
        for (int i = 0; i < NUM_LOGITS; i++)
            logit_buffer[i] <= '0;
    end else begin
        if (read_logits) begin
            wait_for_read <= 1'b1;
            if (wait_for_read && logit_load_count < 10) begin
                logit_buffer[logit_load_count] <= dense_fc_data_out;  // Load logit from dense FC RAM
                logit_load_count <= logit_load_count + 'd1;
            end
        end
    end
end

// Softmax input and output signals
int8_t logits [NUM_LOGITS];                                   // Input logits to softmax
logic signed [31:0] probabilities [NUM_LOGITS];               // Output probabilities from softmax

    softmax_unit #(
        .NUM_CLASSES(10)
        ,.OUTPUT_WIDTH(32) 
        ,.EXP_LUT_ADDR_WIDTH(8)
        ,.EXP_LUT_WIDTH(32) // 256 entries for exponentiation lookup table
        ,.BETA(1)
        ,.INIT_FILE("../rtl/exp_lut.hex") // Initialization file for exponentiation LUT
    ) softmax_layer (
        .clk(clk)
        ,.reset(reset)
        ,.start(softmax_start)
        ,.logits(logits) // Input logits from previous layer
        ,.probabilities(probabilities) // Output probabilities
        ,.valid(softmax_valid) // Indicates softmax output is valid
    );
    
//======================================================================================================
// OUTPUT ROUTING - Final Interface Connections
//======================================================================================================
// This section connects the logit buffer to the softmax input and routes the final
// probabilities to the top-level interface.

// Connect logit buffer to softmax input
always_comb begin
    for (int i = 0; i < NUM_LOGITS; i++) begin
        logits[i] = logit_buffer[i];                          // Route each logit to softmax input
    end
end

// Route final probabilities to top-level output interface
assign probabilities_o = probabilities;                        // Output probabilities to external interface
    
endmodule
