`include "sys_types.svh"
//test harness for patch extractor with padding and sliding window
module tpu_datapath_padded #(
    parameter int MAX_IMG_W = 64,
    parameter int MAX_IMG_H = 64,
    parameter int MAX_CHANNELS = 64,
    parameter int PATCH_SIZE = 4,
    parameter int STRIDE = 1
) (
    input  logic clk,
    input  logic reset,
    
    // Configuration for current processing
    input  logic [$clog2(MAX_IMG_W+1)-1:0] current_img_width,
    input  logic [$clog2(MAX_IMG_H+1)-1:0] current_img_height,
    input  logic [$clog2(MAX_CHANNELS+1)-1:0] current_num_channels,
    
    // Manual control inputs for testing
    input  logic start_patch_generation,
    input  logic start_sliding_window,
    input  logic advance_to_next_patch,
    
    // Status outputs
    output logic [$clog2(MAX_CHANNELS+1)-1:0] current_channel_being_processed,
    output logic [$clog2(MAX_IMG_W)-1:0]      current_patch_col,
    output logic [$clog2(MAX_IMG_H)-1:0]      current_patch_row,
    output logic                               all_channels_done_for_position,
    output logic                               patch_generation_done,
    output logic                               layer_processing_complete,
    
    // Final outputs (int8_t arrays for 4x4 patch)
    output int8_t A0 [0:3],
    output int8_t A1 [0:3], 
    output int8_t A2 [0:3],
    output int8_t A3 [0:3]
);

    // Interconnect signals
    logic patch_extractor_patch_ready;
    logic patch_extractor_extraction_complete;
    logic patch_extractor_next_patch;
    logic [31:0] patch_A0_out, patch_A1_out, patch_A2_out, patch_A3_out;
    logic patch_valid;
    
    // RAM interface
    logic ram_re;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16)-1:0] ram_addr;
    logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3;
    
    // Sliding window outputs (not used in this simple version)
    logic valid_A0, valid_A1, valid_A2, valid_A3, sliding_done;
    
    // Connect status outputs
    assign patch_generation_done = patch_extractor_patch_ready;
    assign layer_processing_complete = patch_extractor_extraction_complete;
    
    // Patch extractor with zero padding
    patch_extractor_with_padding #(
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MAX_CHANNELS(MAX_CHANNELS),
        .PATCH_SIZE(PATCH_SIZE),
        .STRIDE(STRIDE),
        .PAD_LEFT(1),
        .PAD_RIGHT(2),
        .PAD_TOP(1),
        .PAD_BOTTOM(2)
    ) u_patch_extractor (
        .clk(clk),
        .reset(reset),
        
        .img_width(current_img_width),
        .img_height(current_img_height),
        .num_channels(current_num_channels),
        
        .start_patch_extraction(start_patch_generation),
        .next_patch(advance_to_next_patch),
        .patch_ready(patch_extractor_patch_ready),
        .extraction_complete(patch_extractor_extraction_complete),
        
        .current_channel(current_channel_being_processed),
        .current_patch_col(current_patch_col),
        .current_patch_row(current_patch_row),
        .all_channels_done_for_position(all_channels_done_for_position),
        
        .ram_re(ram_re),
        .ram_addr(ram_addr),
        .ram_dout0(ram_dout0),
        .ram_dout1(ram_dout1),
        .ram_dout2(ram_dout2),
        .ram_dout3(ram_dout3),
        
        .patch_A0_out(patch_A0_out),
        .patch_A1_out(patch_A1_out),
        .patch_A2_out(patch_A2_out),
        .patch_A3_out(patch_A3_out),
        .patch_valid(patch_valid)
    );
    
    // Sliding window module (with correct interface)
    sliding_window u_sliding_window (
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
        .done(sliding_done)
    );
    
    // Tensor RAM with correct interface
    tensor_ram #(
        .READ_WIDTH(128),
        .DEPTH_WORDS(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16),
        .WRITE_WIDTH(8),
        .INIT_FILE("../rtl/image_data.hex")
    ) u_tensor_ram (
        .clk(clk),
        .we(1'b0),           // No writes in this test
        .re(ram_re),
        .addr_w('0),         // Not used
        .din(8'h0),          // Not used
        .addr_r(ram_addr),
        .dout(),             // 128-bit output not used
        .dout0(ram_dout0),
        .dout1(ram_dout1),
        .dout2(ram_dout2),
        .dout3(ram_dout3)
    );

endmodule 
