`include "sys_types.svh"

module TPU_Datapath #(
    parameter IMG_W = 32
    ,parameter IMG_H = 32
    ,parameter MAX_N = 64 // Default MAX_N from sta_controller
    ,parameter MAX_NUM_CH = 64
    ,parameter STA_CTRL_N_BITS = $clog2(MAX_N + 1) // Should be 9 bits for MAX_N=512
    ,parameter SA_N = 4  // SA_N from sta_controller
    ,parameter SA_VECTOR_WIDTH = 4 // SA_VECTOR_WIDTH from sta_controller
    ,parameter TOTAL_PES_STA = SA_N * SA_N // Should be 16
    ,parameter NUM_LAYERS = 6 // Number of layers in the model
) (
    input logic clk
    ,input logic reset
    ,input logic read_weights
    ,input logic read_inputs
    ,input logic read_bias
    ,input logic load_bias
    ,input logic incr_weight_ptr
    ,input logic incr_bias_ptr
    ,input logic incr_input_ptr
    ,input logic reset_sta
    ,input logic start
    ,input logic done
    ,input logic reset_ptr_A
    ,input logic reset_ptr_B
    ,input logic reset_ptr_weight
    ,input logic reset_ptr_bias
);

logic done_all_sliding_windows;
// STA Controller signals
logic stall;
logic [1024:0] counter;
// logic                          load_bias;                    // Single load_bias control signal
int32_t                        bias_value;                  // Single bias value to be used for all PEs

logic                          idle;                          // High if STA complex is idle
logic                          array_out_valid;               // High if corresponding val/row/col is valid
logic [127:0]                  array_val_out;                 // Value out of max pool unit
logic [$clog2(MAX_N)-1:0]    array_row_out;                 // Row for corresponding value
logic [$clog2(MAX_N)-1:0]    array_col_out;                 // Column for corresponding value

// logic                          start;       // Pulse to begin sequence
logic					       sta_idle;    // STA signals completion of current tile
// logic                          done;        // Signals completion of current tile computation
logic                          sta_done_computing;
// Current layer and channel index
logic [$clog2(NUM_LAYERS)-1:0] layer_idx; // Index of current layer
logic [$clog2(MAX_NUM_CH)-1:0] chnnl_idx; // Index of current output channel / filter

// Drive STA controller and memory conv parameters
// logic                          reset_sta;
logic [15:0]                   mat_size;
logic                          start_compute;
logic [$clog2(MAX_N)-1:0]      controller_pos_row;
logic [$clog2(MAX_N)-1:0]      controller_pos_col;

// Current layer filter dimensions
logic [$clog2(MAX_NUM_CH+1)-1:0] num_filters;      // Number of output channels (filters) for current layer
logic [$clog2(MAX_NUM_CH+1)-1:0] num_input_channels; // Number of input channels for current layer

// Drive STA controller pool parameters
logic 				           bypass_maxpool;

    // layer_controller layer_controller (
    //     .clk(clk)
    //     ,.reset(reset)

    //     // Inputs to layer_controller
    //     ,.start(start)
    //     ,.stall(stall)
    //     ,.sta_idle(sta_idle)
    //     ,.done(done)
    //     ,.layer_idx(layer_idx)
    //     ,.chnnl_idx(chnnl_idx)
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
    assign layer_idx = 0;

    sta_controller sta_controller (
        .clk(clk)
        ,.reset(reset)
        ,.reset_sta(reset_sta)
        // Inputs to sta_controller
        ,.stall('0)
        ,.bypass_maxpool('0)
        ,.controller_pos_row('0)
        ,.controller_pos_col('0)
        ,.done(done)
        ,.layer_idx('0)
        ,.A0(a0_IN_KERNEL)
        ,.A1(a1_IN_KERNEL)
        ,.A2(a2_IN_KERNEL)
        ,.A3(a3_IN_KERNEL)
        ,.B0(weight_C0)
        ,.B1(weight_C1)
        ,.B2(weight_C2)
        ,.B3(weight_C3)
        ,.load_bias(load_bias)
        ,.bias_value(bias_rom_dout) // Assuming bias_value is a 32-bit value, adjust as needed

        // Outputs from sta_controller
        ,.idle(idle)
        ,.sta_idle(sta_idle)
        ,.array_out_valid(array_out_valid)
        ,.array_val_out(array_val_out)
        ,.array_row_out(array_row_out)
        ,.array_col_out(array_col_out)
    );




    //======================================================================================================

    //Tensor RAM A, and B will be used to store and write , vice versa. never read and write to the same mememory. 
    logic [127:0] tensor_ram_A_dout;
    logic [31:0] tensor_ram_A_dout0;
    logic [31:0] tensor_ram_A_dout1;
    logic [31:0] tensor_ram_A_dout2;
    logic [31:0] tensor_ram_A_dout3;
    logic [127:0] tensor_ram_A_din;
    logic ram_A_we;
    logic ram_A_re;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_A_addr_w;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_A_addr_r;
    // Assuming IMG_W and IMG_H are defined in the module scope, e.g., as parameters or localparams
    // Instantiate tensor_ram for A
    tensor_ram 
    #(
        .READ_WIDTH(128)
        , .DEPTH_WORDS(2048) // Assuming IMG_W and IMG_H are defined in the module scope
        , .WRITE_WIDTH(8)
        , .INIT_FILE("../rtl/image_data.hex") // No initialization file specified
    )
    RAM_A
    (
        .clk(clk)
        ,.we(ram_A_we)
        ,.re(ram_A_re)
        ,.addr_w(ram_A_addr_w)
        ,.din(tensor_ram_A_din)
        ,.addr_r(ram_A_addr_r)
        ,.dout(tensor_ram_A_dout)
        ,.dout0(tensor_ram_A_dout0)
        ,.dout1(tensor_ram_A_dout1)
        ,.dout2(tensor_ram_A_dout2)
        ,.dout3(tensor_ram_A_dout3)
    );


    //======================================================================================================
    //Tensor RAM B, and A will be used to store and write , vice versa. never read and write to the same mememory.
    logic [127:0] tensor_ram_B_dout;
    logic [31:0] tensor_ram_B_dout0;
    logic [31:0] tensor_ram_B_dout1;
    logic [31:0] tensor_ram_B_dout2;
    logic [31:0] tensor_ram_B_dout3;
    logic [127:0] tensor_ram_B_din;
    logic ram_B_we;
    logic ram_B_re;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_B_addr_w;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_B_addr_r;

    
    
    // Instantiate tensor_ram for B
    tensor_ram
    #(
        .READ_WIDTH(128)
        , .DEPTH_WORDS(2048) // Assuming IMG_W and IMG_H are defined in the module scope
        , .WRITE_WIDTH(8)
        , .INIT_FILE("") // No initialization file specified
    )
    RAM_B
    (
        .clk(clk)
        ,.we(ram_B_we)
        ,.re(ram_B_re)
        ,.addr_w(ram_B_addr_w)
        ,.din(tensor_ram_B_din)
        ,.addr_r(ram_B_addr_r)
        ,.dout(tensor_ram_B_dout)
        ,.dout0(tensor_ram_B_dout0)
        ,.dout1(tensor_ram_B_dout1)
        ,.dout2(tensor_ram_B_dout2)
        ,.dout3(tensor_ram_B_dout3)
    );




    // //======================================================================================================
    // logic  incr_ptr_A; // Pointer for reading from tensor_ram A
    // pointer 
    // #(
    //     .DEPTH(IMG_W * IMG_H) // Assuming IMG_W and IMG_H are defined in the module scope
    // )
    // pixel_pointer_A
    // (
    //     .clk(clk)
    //     ,.reset(reset)
    //     ,.reset_ptr(reset_ptr_A)
    //     ,.rst_ptr_val(0)
    //     ,.incr_ptr(incr_ptr_A) // Assuming this is the control signal to increment the pointer
    //     ,.ptr(ram_A_addr_r) // Output pointer for reading from tensor_ram A
    // );

    // logic incr_ptr_B; // Pointer for reading from tensor_ram B
    // pointer
    // #(
    //     .DEPTH(IMG_W * IMG_H) // Assuming IMG_W and IMG_H are defined in the module scope
    // )
    // pixel_pointer_B
    // (
    //     .clk(clk)
    //     ,.reset(reset)
    //     ,.reset_ptr(reset_ptr_B)
    //     ,.rst_ptr_val(0)
    //     ,.incr_ptr(incr_ptr_B) // Assuming this is the control signal to increment the pointer
    //     ,.ptr(ram_B_addr_r) // Output pointer for reading from tensor_ram B
    // );




    //======================================================================================================
    // Unified Buffer for spatial block extraction with channel iteration
    //current layer input image dimensions
    logic [$clog2(64+1)-1:0] img_width;
    logic [$clog2(64+1)-1:0] img_height;
    logic [$clog2(64+1)-1:0] num_channels;
    //unified buffer control signals
    logic start_block_extraction;
    logic next_channel_group;
    logic next_spatial_block;
    logic block_ready;
    logic block_extraction_complete;  
    logic all_channels_done;
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
    logic patches_valid;

    unified_buffer 
    #(
        .MAX_IMG_W(64)
        ,.MAX_IMG_H(64)
        ,.BUFFER_SIZE(7)
        ,.PATCH_SIZE(4)
        ,.PATCHES_PER_BLOCK(4)
        ,.MAX_CHANNELS(64)
        ,.MAX_PADDING(3)  // Support up to 3 pixels of padding
    )
    UNIFIED_BUFFER
    (
        .clk(clk)
        ,.reset(reset)
        ,.start_extraction(start_block_extraction)
        ,.next_channel_group(next_channel_group)
        ,.next_spatial_block(next_spatial_block)
        ,.block_ready(block_ready)
        ,.extraction_complete(block_extraction_complete)
        ,.all_channels_done(all_channels_done)
        ,.img_width(img_width)
        ,.img_height(img_height)
        ,.num_channels(num_channels)
        ,.pad_top(3'd1)      // 1 pixel top padding (configurable)
        ,.pad_bottom(3'd1)   // 1 pixel bottom padding 
        ,.pad_left(3'd1)     // 1 pixel left padding
        ,.pad_right(3'd1)    // 1 pixel right padding
        ,.ram_re(buffer_ram_read_enable)
        ,.ram_addr(buffer_ram_addr)
        ,.ram_dout0(buffer_ram_in0)
        ,.ram_dout1(buffer_ram_in1)
        ,.ram_dout2(buffer_ram_in2)
        ,.ram_dout3(buffer_ram_in3)
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
    // Spatial Data Formatter - converts unified buffer patches to systolic array format
    logic start_spatial_formatting;
    logic next_col_request;
    logic formatted_data_valid;
    logic all_cols_sent;
    int8_t formatted_A0 [0:3]; // Row 0, 4 channels of current column position
    int8_t formatted_A1 [0:3]; // Row 1, 4 channels of current column position  
    int8_t formatted_A2 [0:3]; // Row 2, 4 channels of current column position
    int8_t formatted_A3 [0:3]; // Row 3, 4 channels of current column position

    spatial_data_formatter SPATIAL_FORMATTER
    (
        .clk(clk)
        ,.reset(reset)
        ,.start_formatting(start_spatial_formatting)
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
        ,.formatted_data_valid(formatted_data_valid)
        ,.all_cols_sent(all_cols_sent)
    );

    //======================================================================================================
    // SLIDING WINDOW for input images - now receives formatted spatial data
    logic start_sliding_window; // Control signal to start the sliding window operation
    logic valid_in_sliding_window; // Valid signal for the sliding window input
    // int8_t a0_IN_KERNEL [3:0]; // Now comes from spatial formatter
    // int8_t a1_IN_KERNEL [3:0]; // Now comes from spatial formatter  
    // int8_t a2_IN_KERNEL [3:0]; // Now comes from spatial formatter
    // int8_t a3_IN_KERNEL [3:0]; // Now comes from spatial formatter
    logic valid_IN_KERNEL_A0;
    logic valid_IN_KERNEL_A1;
    logic valid_IN_KERNEL_A2;
    logic valid_IN_KERNEL_A3;

    // Connect formatted data directly to systolic array inputs
    assign a0_IN_KERNEL = formatted_A0;
    assign a1_IN_KERNEL = formatted_A1;
    assign a2_IN_KERNEL = formatted_A2;
    assign a3_IN_KERNEL = formatted_A3;
    assign valid_in_sliding_window = formatted_data_valid;

    logic done_IN_KERNEL; // Signal indicating the sliding window operation is done
    sliding_window IN_KERNELS
    (
        .clk(clk)
        ,.reset(reset|reset_sta)
        ,.start(start_sliding_window)
        ,.valid_in(valid_in_sliding_window)
        ,.A0_in({formatted_A0[3], formatted_A0[2], formatted_A0[1], formatted_A0[0]}) // Pack array into 32-bit
        ,.A1_in({formatted_A1[3], formatted_A1[2], formatted_A1[1], formatted_A1[0]}) // Pack array into 32-bit
        ,.A2_in({formatted_A2[3], formatted_A2[2], formatted_A2[1], formatted_A2[0]}) // Pack array into 32-bit
        ,.A3_in({formatted_A3[3], formatted_A3[2], formatted_A3[1], formatted_A3[0]}) // Pack array into 32-bit
        ,.A0(a0_IN_KERNEL)
        ,.A1(a1_IN_KERNEL)
        ,.A2(a2_IN_KERNEL)
        ,.A3(a3_IN_KERNEL)
        ,.valid_A0(valid_IN_KERNEL_A0)
        ,.valid_A1(valid_IN_KERNEL_A1)
        ,.valid_A2(valid_IN_KERNEL_A2)
        ,.valid_A3(valid_IN_KERNEL_A3)
        ,.done(done_IN_KERNEL)
    );



    //======================================================================================================




    //instantiate WEIGHT ROM

    logic WEIGHT_READ_EN;
    logic [($clog2(16384))-1:0] weight_rom_addr; // Assuming 16384 is the depth of the weight ROM
     int32_t weight_rom_dout0; // Output data from the weight ROM
     int32_t weight_rom_dout1; // Output data from the weight ROM
     int32_t weight_rom_dout2; // Output data from the weight ROM
     int32_t weight_rom_dout3; // Output data from the weight ROM

    assign WEIGHT_READ_EN = read_weights; // Control signal to enable reading from weight ROM
    // ROM for weights, assuming 16 8-bit values per read
    weight_rom 
    #(
        .WIDTH(128) // 4 8-bit pixels per read/write
        , .DEPTH(16384) // Assuming IMG_W and IMG_H are defined in the module scope
        , .INIT_FILE("../fakemodel/tflite_conv_kernel_weights.hex") // No initialization file specified
    )
    WEIGHT_ROM
    (
        .clk(clk)
        ,.read_enable(WEIGHT_READ_EN)
        ,.addr(weight_rom_addr)
        ,.data0(weight_rom_dout0)
        ,.data1(weight_rom_dout1)
        ,.data2(weight_rom_dout2)
        ,.data3(weight_rom_dout3)
    );

    logic INCR_WEIGHT_PTR; // Control signal to increment the weight pointer
    // logic reset_ptr_weight;
    pointer weight_pointer
     (
        .clk(clk)
        ,.reset(reset)
        ,.reset_ptr(reset_ptr_weight)
        ,.rst_ptr_val(0)
        ,.incr_ptr(INCR_WEIGHT_PTR) // Assuming this is the control signal to increment the pointer
        ,.ptr(weight_rom_addr) // Output pointer for reading from weight_rom
     );

    //======================================================================================================

    logic valid_in_sliding_window_weights; // Valid signal for the sliding window input for weights

     int8_t weight_C0 [0:3]; // Assuming 4 input channels for the sliding window weights
     int8_t weight_C1 [0:3]; // Assuming 4 input channels for the sliding window weights
     int8_t weight_C2 [0:3]; // Assuming 4 input channels for the sliding window weights
     int8_t weight_C3 [0:3]; // Assuming 4 input channels for the sliding window weights
    logic valid_C0; // Valid signal for weight_C0
    logic valid_C1; // Valid signal for weight_C1
    logic valid_C2; // Valid signal for weight_C2
    logic valid_C3; // Valid signal for weight_C3
    logic done_WEIGHTS; // Signal indicating the sliding window operation for weights is done
    sliding_window WEIGHTS_IN
    (  
        .clk(clk)
        ,.reset(reset|reset_sta)
        ,.start(start_sliding_window)
        ,.valid_in(valid_in_sliding_window_weights)
        ,.A0_in(weight_rom_dout0)
        ,.A1_in(weight_rom_dout1)
        ,.A2_in(weight_rom_dout2)
        ,.A3_in(weight_rom_dout3)
        ,.A0(weight_C0)
        ,.A1(weight_C1)
        ,.A2(weight_C2)
        ,.A3(weight_C3)
        ,.valid_A0(valid_C0)
        ,.valid_A1(valid_C1)
        ,.valid_A2(valid_C2)
        ,.valid_A3(valid_C3)
        ,.done(done_WEIGHTS)
    );

    //======================================================================================================
    //Bias ROM

    logic BIAS_READ_EN;
    logic [($clog2(256))-1:0] bias_rom_addr;
    logic [31:0] bias_rom_dout;

    assign BIAS_READ_EN = read_bias;
    bias_rom
    #(
        .WIDTH(32)
        ,.DEPTH(256)
        ,.INIT_FILE("../fakemodel/tflite_bias_weights.hex")
    )
    BIAS_ROM
    (
        .clk(clk)
        ,.read_enable(BIAS_READ_EN)
        ,.addr(bias_rom_addr)
        ,.bias_out(bias_rom_dout)
    );

    logic INCR_BIAS_PTR;
    // logic reset_ptr_bias;
    pointer 
    #(
        .DEPTH(256)
    )
    BIAS_POINTER
    (
        .clk(clk)
        ,.reset(reset)
        ,.reset_ptr(reset_ptr_bias)
        ,.rst_ptr_val(0)
        ,.incr_ptr(INCR_BIAS_PTR)
        ,.ptr(bias_rom_addr)
    );

     //======================================================================================================

    // Control signals for writing and reading if layer index lsb is 0 or 1
    assign ram_A_we = layer_idx[0] & array_out_valid; // Example logic to control write enable for tensor_ram A based on layer_idx
    assign ram_B_we = ~layer_idx[0] & array_out_valid; // Example logic to control write enable for tensor_ram B based on layer_idx
    assign ram_A_re = ~layer_idx[0] ? buffer_ram_read_enable : 1'b0;
    assign ram_B_re = layer_idx[0] ? buffer_ram_read_enable : 1'b0;
    assign ram_A_addr_r = ~layer_idx[0] ? buffer_ram_addr : 0;
    assign ram_B_addr_r = layer_idx[0] ? buffer_ram_addr : 0;
    always_comb begin
        buffer_ram_in0 = layer_idx[0] ? tensor_ram_B_dout0 : tensor_ram_A_dout0; // Example logic to select input for sliding window A0
        buffer_ram_in1 = layer_idx[0] ? tensor_ram_B_dout1 : tensor_ram_A_dout1; // Example logic to select input for sliding window A1
        buffer_ram_in2 = layer_idx[0] ? tensor_ram_B_dout2 : tensor_ram_A_dout2; // Example logic to select input for sliding window A2
        buffer_ram_in3 = layer_idx[0] ? tensor_ram_B_dout3 : tensor_ram_A_dout3; // Example logic to select input for sliding window A3

        if(layer_idx[0] == 0) begin
            tensor_ram_B_din = array_out_valid ? array_val_out : 128'b0;
            tensor_ram_A_din = 128'b0;
        end else begin
            tensor_ram_A_din = array_out_valid ? array_val_out : 128'b0;
            tensor_ram_B_din = 128'b0;
        end
    end

    always_ff @(posedge clk) begin
        if(layer_idx[0] == 0 && array_out_valid) begin
            ram_A_addr_w <= ram_A_addr_w+1;
            ram_B_addr_w <= 0;
        end else if(layer_idx[0] == 1 && array_out_valid) begin
            ram_A_addr_w <= 0;
            ram_B_addr_w <= ram_B_addr_w+1;
        end
    end    
        


     
    

    //======================================================================================================

    // Control Logic for Spatial Processing
    assign start_block_extraction = start; 

    // Add loading sequence control
    logic buffer_loading_complete;
    logic buffer_data_valid;

    // Ensure RAM data is valid before unified buffer starts
    always_ff @(posedge clk) begin
        if (reset) begin
            buffer_data_valid <= 1'b0;
        end else begin
            // RAM data is valid when we have valid addresses and read enable
            buffer_data_valid <= (ram_A_re | ram_B_re) & 
                               (layer_idx[0] ? (ram_B_addr_r < (IMG_W * IMG_H)) : 
                                             (ram_A_addr_r < (IMG_W * IMG_H)));
        end
    end

    // Track buffer loading completion
    always_ff @(posedge clk) begin
        if (reset) begin
            buffer_loading_complete <= 1'b0;
        end else begin
            // Buffer is fully loaded when we've loaded all required data
            buffer_loading_complete <= block_ready & buffer_data_valid;
        end
    end

    // Only start spatial formatting when buffer is fully loaded
    assign start_spatial_formatting = buffer_loading_complete;

    // Improve RAM address calculation
    always_comb begin
        // Calculate base address based on layer index
        logic [$clog2(IMG_W * IMG_H)-1:0] base_addr;
        base_addr = layer_idx[0] ? ram_B_addr_r : ram_A_addr_r;
        
        // Calculate padded coordinates
        logic [$clog2(IMG_W + 2*MAX_PADDING)-1:0] padded_row, padded_col;
        padded_row = base_addr / IMG_W;
        padded_col = base_addr % IMG_W;
        
        // Check if we're in padding region
        logic in_padding;
        in_padding = (padded_row < pad_top) || 
                     (padded_row >= (IMG_H + pad_top)) ||
                     (padded_col < pad_left) || 
                     (padded_col >= (IMG_W + pad_left));
        
        // Only access RAM for non-padding regions
        if (!in_padding) begin
            // Convert to actual image coordinates
            logic [$clog2(IMG_W)-1:0] img_row, img_col;
            img_row = padded_row - pad_top;
            img_col = padded_col - pad_left;
            
            // Calculate final RAM address
            buffer_ram_addr = (img_row * IMG_W + img_col) * (num_channels / 4);
        end else begin
            // For padding regions, use a safe address that won't corrupt data
            buffer_ram_addr = '0;
        end
    end

    assign next_channel_group = all_channels_done; // Request next channel group when all channels are processed
    assign next_spatial_block = all_cols_sent; // Request next spatial block when all columns are sent
    assign next_col_request = sta_idle; // Request next column when STA completes current column
    
    assign start_sliding_window = start_spatial_formatting; 
    assign INCR_WEIGHT_PTR = incr_weight_ptr;
    assign INCR_BIAS_PTR = incr_bias_ptr;
    
    assign done_all_sliding_windows = done_IN_KERNEL & done_WEIGHTS;
    assign sta_done_computing = all_cols_sent & sta_idle; // Complete when all spatial columns processed

    //======================================================================================================
endmodule
