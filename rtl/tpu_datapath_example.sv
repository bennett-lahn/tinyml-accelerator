`include "sys_types.svh"

// Example integration showing how to use patch_extractor with tensor_ram and sliding_window
// Patch extractor now iterates: all channels for position (0,0), then all channels for position (0,1), etc.
// Memory layout is channel-last: [row, col, channel]
module tpu_datapath_example #(
    parameter int IMG_W = 32,
    parameter int IMG_H = 32,
    parameter int MAX_CHANNELS = 64
) (
    input  logic                        clk,
    input  logic                        reset,
    
    // Layer configuration
    input  logic [$clog2(IMG_W+1)-1:0] current_img_width,
    input  logic [$clog2(IMG_H+1)-1:0] current_img_height, 
    input  logic [$clog2(MAX_CHANNELS+1)-1:0] current_num_channels,
    
    // Control signals - now with manual control
    input  logic                        start_patch_generation,      // Manual trigger for patch generation
    input  logic                        start_sliding_window,        // Manual trigger for sliding window
    input  logic                        advance_to_next_patch,       // Next channel or next position
    output logic                        patch_generation_done,       // Patch is ready/complete
    output logic                        sliding_window_done,         // Sliding window output complete
    output logic                        layer_processing_complete,
    
    // Status outputs
    output logic [$clog2(MAX_CHANNELS+1)-1:0] current_channel_being_processed,
    output logic [$clog2(IMG_W)-1:0]          current_patch_col,
    output logic [$clog2(IMG_H)-1:0]          current_patch_row,
    output logic                               all_channels_done_for_position,
    
    // Outputs to systolic array
    output int8_t                       A0 [0:3],
    output int8_t                       A1 [0:3], 
    output int8_t                       A2 [0:3],
    output int8_t                       A3 [0:3],
    output logic                        valid_A0,
    output logic                        valid_A1,
    output logic                        valid_A2,
    output logic                        valid_A3
);

    // Signals between patch_extractor and tensor_ram
    logic                            ram_re;
    logic [$clog2(IMG_W*IMG_H*MAX_CHANNELS/16)-1:0] ram_addr;
    logic [31:0]                     ram_dout0, ram_dout1, ram_dout2, ram_dout3;
    
    // Signals between patch_extractor and sliding_window
    logic [31:0]                     patch_A0_out, patch_A1_out, patch_A2_out, patch_A3_out;
    logic                            patch_valid;
    logic                            patch_ready;
    logic                            extraction_complete;
    
    // Sliding window control
    logic                            sliding_window_start;
    
    // Instantiate tensor_ram with channel-last layout
    // Memory depth needs to accommodate width * height * channels pixels
    // Each 128-bit word contains 16 pixels
    tensor_ram #(
        .READ_WIDTH(128),
        .DEPTH_WORDS(IMG_W * IMG_H * MAX_CHANNELS / 16),
        .WRITE_WIDTH(8),
        .INIT_FILE("../rtl/image_data.hex")
    ) input_tensor_ram (
        .clk(clk),
        .we(1'b0),  // Read-only for input data
        .re(ram_re),
        .addr_w('0),
        .din(8'h0),
        .addr_r(ram_addr),
        .dout(),    // Not used, we use the split outputs
        .dout0(ram_dout0),
        .dout1(ram_dout1),
        .dout2(ram_dout2),
        .dout3(ram_dout3)
    );
    
    // Instantiate patch_extractor
    patch_extractor #(
        .MAX_IMG_W(IMG_W),
        .MAX_IMG_H(IMG_H),
        .MAX_CHANNELS(MAX_CHANNELS),
        .PATCH_SIZE(4),
        .STRIDE(1)
    ) patch_ext (
        .clk(clk),
        .reset(reset),
        
        // Configuration
        .img_width(current_img_width),
        .img_height(current_img_height),
        .num_channels(current_num_channels),
        
        // Control
        .start_patch_extraction(start_patch_generation),
        .next_patch(advance_to_next_patch),
        .patch_ready(patch_ready),
        .extraction_complete(extraction_complete),
        
        // Status outputs
        .current_channel(current_channel_being_processed),
        .current_patch_col(current_patch_col),
        .current_patch_row(current_patch_row),
        .all_channels_done_for_position(all_channels_done_for_position),
        
        // Interface to tensor_ram
        .ram_re(ram_re),
        .ram_addr(ram_addr),
        .ram_dout0(ram_dout0),
        .ram_dout1(ram_dout1),
        .ram_dout2(ram_dout2),
        .ram_dout3(ram_dout3),
        
        // Interface to sliding_window
        .patch_A0_out(patch_A0_out),
        .patch_A1_out(patch_A1_out),
        .patch_A2_out(patch_A2_out),
        .patch_A3_out(patch_A3_out),
        .patch_valid(patch_valid)
    );
    
    // Instantiate sliding_window
    sliding_window input_sliding_window (
        .clk(clk),
        .reset(reset),
        .start(start_sliding_window),
        .valid_in(patch_valid),
        .A0_in(patch_A0_out),
        .A1_in(patch_A1_out),
        .A2_in(patch_A2_out),
        .A3_in(patch_A3_out),
        .A0(A0),
        .A1(A1),
        .A2(A2),
        .A3(A3),
        .valid_A0(valid_A0),
        .valid_A1(valid_A1),
        .valid_A2(valid_A2),
        .valid_A3(valid_A3),
        .done(sliding_window_done)
    );
    
    // Control logic - now manual
    assign patch_generation_done = patch_ready;  // Patch ready when extractor completes
    assign layer_processing_complete = extraction_complete;

endmodule 
