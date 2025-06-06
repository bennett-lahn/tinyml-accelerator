`include "sys_types.svh"

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

logic                          stall;
// logic                          load_bias;                    // Single load_bias control signal
int32_t                        bias_value;                   // Single bias value to be used for all PEs

logic                          idle;                         // High if STA complex is idle
logic                          array_out_valid;              // High if corresponding val/row/col is valid
int8_t                         array_val_out;                // Value out of max pool unit
logic [$clog2(MAX_N)-1:0]      array_row_out;                // Row for corresponding value
logic [$clog2(MAX_N)-1:0]      array_col_out;                // Column for corresponding value

// logic                          start;                        // Pulse to begin sequence
// logic                          done;                         // Signals completion of current tile computation

// Drive STA controller and memory conv parameters
// logic                          reset_sta;
logic [15:0]                   mat_size;
logic                          start_compute;
logic [$clog2(MAX_N)-1:0]      array_index_out;

// Current layer filter dimensions
logic [$clog2(MAX_NUM_CH+1)-1:0] num_filters;        // Number of output channels (filters) for current layer
logic [$clog2(MAX_NUM_CH+1)-1:0] num_input_channels; // Number of input channels for current layer

// Drive STA controller pool parameters
logic                            bypass_valid;
logic [BYPASS_IDX_BITS-1:0]      bypass_index;
int32_t                          bypass_value;
logic [$clog2(MAX_N)-1:0]        controller_pos_row;
logic [$clog2(MAX_N)-1:0]        controller_pos_col;

logic [$clog2(MAX_N)-1:0] block_start_col_addr; // Write address for tensor_ram A
logic [$clog2(MAX_N)-1:0] block_start_row_addr; // Write address for tensor_ram B

assign controller_pos_row = block_start_row_addr; // Use block start row address for controller position
assign controller_pos_col = block_start_col_addr; // Use block start column address for controller position

int8_t B0 [SA_VECTOR_WIDTH];
int8_t B1 [SA_VECTOR_WIDTH];
int8_t B2 [SA_VECTOR_WIDTH];
int8_t B3 [SA_VECTOR_WIDTH];

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
// Bias ROM
//======================================================================================================

bias_rom #(
    .WIDTH(ROM_BIAS_WIDTH)
    ,.DEPTH(ROM_BIAS_DEPTH)
    ,.MAX_NUM_CH(MAX_NUM_CH)
    ,.INIT_FILE("../fakemodel/tflite_bias_weights.hex")
) BIAS_ROM (
    .clk(clk)
    ,.read_enable(read_bias)
    ,.layer_idx(layer_idx)
    ,.channel_idx(channel_idx)
    ,.bias_out(bias_value)
);

//======================================================================================================
// TENSOR RAM
//======================================================================================================

//Tensor RAM A, and B will be used to store and write , vice versa. never read and write to the same mememory. 
int32_t tensor_ram_A_dout0;
int32_t tensor_ram_A_dout1;
int32_t tensor_ram_A_dout2;
int32_t tensor_ram_A_dout3;
int8_t tensor_ram_A_din;
logic tensor_ram_A_data_valid;
logic ram_A_we;
logic ram_A_re;
logic [($clog2(RAM_WORD_DEPTH*(RAM_READ_WIDTH/RAM_WRITE_WIDTH)))-1:0] ram_A_addr_r;

// Instantiate tensor_ram A
tensor_ram #(
    .INIT_FILE("../rtl/image_data.hex")
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

int32_t tensor_ram_B_dout0;
int32_t tensor_ram_B_dout1;
int32_t tensor_ram_B_dout2;
int32_t tensor_ram_B_dout3;
int8_t tensor_ram_B_din;
logic tensor_ram_B_data_valid;
logic ram_B_we;
logic ram_B_re;
logic [($clog2(RAM_WORD_DEPTH*(RAM_READ_WIDTH/RAM_WRITE_WIDTH)))-1:0] ram_B_addr_r;

// Instantiate tensor_ram B
tensor_ram #(
    .INIT_FILE("")
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

// Control signals for writing and reading if layer index lsb is 0 or 1
assign ram_A_we = layer_idx[0] & array_out_valid; // Example logic to control write enable for tensor_ram A based on layer_idx
assign ram_B_we = ~layer_idx[0] & array_out_valid; // Example logic to control write enable for tensor_ram B based on layer_idx
assign ram_A_re = flatten_stage ? request_chunk : ~layer_idx[0] ? buffer_ram_read_enable : 1'b0;
assign ram_B_re = layer_idx[0] ? buffer_ram_read_enable : 1'b0;
assign ram_A_addr_r = flatten_stage? chunk_addr : ~layer_idx[0] ? buffer_ram_addr : 0;
assign ram_B_addr_r = layer_idx[0] ? buffer_ram_addr : 0;

logic ram_data_valid;
assign ram_data_valid = tensor_ram_A_data_valid | tensor_ram_B_data_valid;

always_comb begin
    buffer_ram_in0 = layer_idx[0] ? tensor_ram_B_dout0 : tensor_ram_A_dout0; // Example logic to select input for sliding window A0
    buffer_ram_in1 = layer_idx[0] ? tensor_ram_B_dout1 : tensor_ram_A_dout1; // Example logic to select input for sliding window A1
    buffer_ram_in2 = layer_idx[0] ? tensor_ram_B_dout2 : tensor_ram_A_dout2; // Example logic to select input for sliding window A2
    buffer_ram_in3 = layer_idx[0] ? tensor_ram_B_dout3 : tensor_ram_A_dout3; // Example logic to select input for sliding window A3

    if(layer_idx[0] == 0) begin
        tensor_ram_B_din = (array_out_valid) ? array_val_out : 'd0;
        tensor_ram_A_din = 'd0;
    end else begin
        tensor_ram_A_din = (array_out_valid) ? array_val_out : 'd0;
        tensor_ram_B_din = 'd0;
    end
end   

//======================================================================================================
// UNIFIED BUFFER AND ACTIVATION DATAPATH
//======================================================================================================

// Unified Buffer for spatial block extraction with channel iteration
//current layer input image dimensions

//unified buffer read enable and address
logic buffer_ram_read_enable;
logic [$clog2(64*64*64/4)-1:0] buffer_ram_addr;
//unified buffer input from tensor RAM
logic [31:0] buffer_ram_in0;
logic [31:0] buffer_ram_in1;
logic [31:0] buffer_ram_in2;
logic [31:0] buffer_ram_in3;
//unified buffer output - All 7x7 positions for spatial streaming
logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out, patch_pe04_out, patch_pe05_out, patch_pe06_out;
logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out, patch_pe14_out, patch_pe15_out, patch_pe16_out;
logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out, patch_pe24_out, patch_pe25_out, patch_pe26_out;
logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out, patch_pe34_out, patch_pe35_out, patch_pe36_out;
logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out, patch_pe44_out, patch_pe45_out, patch_pe46_out;
logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out, patch_pe54_out, patch_pe55_out, patch_pe56_out;
logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out, patch_pe64_out, patch_pe65_out, patch_pe66_out;

// Padding control signals
// logic [$clog2(MAX_PADDING+1)-1:0] pad_top;
// logic [$clog2(MAX_PADDING+1)-1:0] pad_bottom;
// logic [$clog2(MAX_PADDING+1)-1:0] pad_left;
// logic [$clog2(MAX_PADDING+1)-1:0] pad_right;


unified_buffer_harness #(
    .MAX_IMG_W(IMG_W)
    ,.MAX_IMG_H(IMG_H)
    ,.MAX_CHANNELS(MAX_NUM_CH)
    ,.MAX_PADDING(MAX_PADDING)
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

// Spatial Data Formatter - converts unified buffer patches to systolic array format

int8_t formatted_A0 [0:3]; // Row 0, 4 channels of current column position
int8_t formatted_A1 [0:3]; // Row 1, 4 channels of current column position  
int8_t formatted_A2 [0:3]; // Row 2, 4 channels of current column position
int8_t formatted_A3 [0:3]; // Row 3, 4 channels of current column position
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
// WEIGHT DATAPATH AND ROM
//======================================================================================================

    logic weight_rom_read_enable;
    logic [$clog2(ROM_WEIGHT_WIDTH)-1:0] weight_rom_addr;
    int32_t weight_rom_dout0;
    int32_t weight_rom_dout1;
    int32_t weight_rom_dout2;
    int32_t weight_rom_dout3;

    logic weight_load_idle;
    logic weight_load_complete;

    weight_loader # (
        .MAX_NUM_CH(MAX_NUM_CH)
        ,.SA_N(SA_N)
        ,.VECTOR_WIDTH(SA_VECTOR_WIDTH)
        ,.ROM_DEPTH(ROM_WEIGHT_DEPTH)
    ) weight_loader (
        .clk(clk)
        ,.reset(reset | reset_datapath | reset_sta)
        ,.start(start & patches_valid)
        ,.stall(stall)
        
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

    // logic [($clog2(ROM_WEIGHT_DEPTH))-1:0] weight_rom_addr; 
    // int32_t weight_rom_dout0; // Output data from the weight ROM
    // int32_t weight_rom_dout1; // Output data from the weight ROM
    // int32_t weight_rom_dout2; // Output data from the weight ROM
    // int32_t weight_rom_dout3; // Output data from the weight ROM

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
// FLATTEN
//======================================================================================================


logic chunk_valid; // Indicates if the current chunk of input data is valid
logic [127:0] input_chunk; // 128-bit input chunk (16 int8_t values)
logic request_chunk; // Request next 128-bit chunk
logic [$clog2(64)-1:0] chunk_addr; // Current chunk address (0-63)
logic output_read_enable; // Enable reading output data
logic [31:0] input_data [0:64-1]; // Input data from previous layer
logic [$clog2(2)-1:0] input_row;
logic [$clog2(2)-1:0] input_col;

int8_t output_data_flatten; // Output data (one element at a time)
logic [$clog2(256)-1:0] output_addr_flatten; // Current output address
logic output_valid; // Indicates output data is valid
logic all_outputs_sent; // Indicates all data has been output


flatten_layer #(
    .INPUT_HEIGHT(IMG_H)
    ,.INPUT_WIDTH(IMG_W)
    ,.INPUT_CHANNELS(MAX_NUM_CH)
    ,.CHUNK_SIZE(16) // 128 bits = 16 int8_t values
    ,.TOTAL_CHUNKS((IMG_H * IMG_W * MAX_NUM_CH) / 16) // Total chunks based on input dimensions
    ,.OUTPUT_SIZE(IMG_H * IMG_W * MAX_NUM_CH) // Total output size
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
        ,.INIT_FILE("../fakemodel/tflite_dense_fc_weights.hex") // Initialization file for weights
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
        ,.DEPTH(72) // 72 words for the fully connected layer
        ,.INIT_FILE("../fakemodel/tflite_fc_bias_weights.hex") // Initialization file for bias weights
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
//DENSE WEIGHT ROM
//======================================================================================================

    logic dense_weight_rom_read_enable;
    logic [$clog2(256*64)-1:0] dense_weight_rom_addr; // Address for weight ROM
    logic [7:0] dense_weight_rom_dout; // Output data from the weight ROM

    dense_weight_rom #(
        .WIDTH(8) // 8 bits per word
        ,.DEPTH(256*64) // 256 input channels, 64 output channels
        ,.INIT_FILE("../fakemodel/tflite_dense_weights.hex") // Initialization file for weights
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

    logic bypass_maxpool;
    logic bypass_relu;

    assign bypass_maxpool = (layer_idx > 'd3);
    assign bypass_relu = (layer_idx == 'd5);

    assign fc_bias_rom_read_enable = bias_rom_re;
    assign fc_bias_rom_addr = bias_rom_addr;
    assign bias_rom_dout = fc_bias_rom_dout;
    assign dense_weight_rom_read_enable = weight_rom_re;
    assign dense_weight_rom_addr = weight_rom_addr_dense;
    assign weight_rom_dout = dense_weight_rom_dout;
    assign dense_fc_read_enable = tensor_ram_re;
    assign dense_fc_read_addr = tensor_ram_addr;
    // assign dense_fc_write_addr = array_index_out;
    // assign dense_fc_data_in = array_val_out; 
    assign tensor_ram_dout = dense_fc_data_out;
    assign bypass_valid = output_ready;
    assign bypass_value = output_data;
    assign bypass_index = output_addr; // Assuming output_addr is used for bypass index


//======================================================================================================
// SOFTMAX
//======================================================================================================

    localparam int NUM_LOGITS = 10;
    logic [$clog2(NUM_LOGITS)-1:0] logit_load_count;
    int8_t logit_buffer [NUM_LOGITS];
    logic wait_for_read; // Used to wait one cycle for initial fc_ram load delay

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
                    logit_buffer[logit_load_count] <= dense_fc_data_out;
                    logit_load_count <= logit_load_count + 'd1;
                end
            end
        end
    end

    int8_t logits [NUM_LOGITS];
    logic signed [31:0] probabilities [NUM_LOGITS]; // Output data from softmax layer

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
    
    // Connect logit_buffer to logits input
    always_comb begin
        for (int i = 0; i < NUM_LOGITS; i++) begin
            logits[i] = logit_buffer[i];
        end
    end
    
    assign probabilities_o = probabilities; // Output probabilities to top-level interface
    
endmodule
