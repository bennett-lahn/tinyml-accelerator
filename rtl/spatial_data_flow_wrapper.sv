`include "sys_types.svh"

module spatial_data_flow_wrapper (
    input logic clk,
    input logic reset,
    
    // Unified Buffer Control Signals
    input logic start_extraction,
    input logic next_channel_group,
    input logic next_spatial_block,
    output logic block_ready,
    output logic extraction_complete,
    output logic all_channels_done,
    
    // Image parameters
    input logic [$clog2(64+1)-1:0] img_width,
    input logic [$clog2(64+1)-1:0] img_height,
    input logic [$clog2(64+1)-1:0] num_channels,
    
    // Zero padding parameters
    input logic [$clog2(3+1)-1:0] pad_top,     // Top padding
    input logic [$clog2(3+1)-1:0] pad_bottom,  // Bottom padding  
    input logic [$clog2(3+1)-1:0] pad_left,    // Left padding
    input logic [$clog2(3+1)-1:0] pad_right,   // Right padding
    
    // RAM interface
    output logic ram_re,
    output logic [$clog2(64*64*64/4)-1:0] ram_addr,
    input logic [31:0] ram_dout0,
    input logic [31:0] ram_dout1,
    input logic [31:0] ram_dout2,
    input logic [31:0] ram_dout3,
    
    // Spatial Formatter Control
    input logic start_formatting,
    output logic formatted_data_valid,
    output logic all_cols_sent,
    output logic next_block,
    
    // Spatial Formatter Outputs
    output int8_t formatted_A0 [0:3],
    output int8_t formatted_A1 [0:3],
    output int8_t formatted_A2 [0:3],
    output int8_t formatted_A3 [0:3]
);

    // Internal signals connecting unified buffer to spatial formatter
    logic patches_valid;
    
    // All 7x7 patch outputs from unified buffer
    logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out, patch_pe04_out, patch_pe05_out, patch_pe06_out;
    logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out, patch_pe14_out, patch_pe15_out, patch_pe16_out;
    logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out, patch_pe24_out, patch_pe25_out, patch_pe26_out;
    logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out, patch_pe34_out, patch_pe35_out, patch_pe36_out;
    logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out, patch_pe44_out, patch_pe45_out, patch_pe46_out;
    logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out, patch_pe54_out, patch_pe55_out, patch_pe56_out;
    logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out, patch_pe64_out, patch_pe65_out, patch_pe66_out;

    // Instantiate Unified Buffer
    unified_buffer #(
        .MAX_IMG_W(64),
        .MAX_IMG_H(64),
        .BUFFER_SIZE(7),
        .PATCH_SIZE(4),
        .PATCHES_PER_BLOCK(4),
        .MAX_CHANNELS(64),
        .MAX_PADDING(3)  // Support up to 3 pixels of padding
    ) u_unified_buffer (
        .clk(clk),
        .reset(reset),
        .start_extraction(start_extraction),
        .next_channel_group(next_channel_group),
        .next_spatial_block(next_spatial_block),
        .block_ready(block_ready),
        .extraction_complete(extraction_complete),
        .all_channels_done(all_channels_done),
        .img_width(img_width),
        .img_height(img_height),
        .num_channels(num_channels),
        .pad_top(pad_top),       // Connect padding parameters
        .pad_bottom(pad_bottom),
        .pad_left(pad_left),
        .pad_right(pad_right),
        .ram_re(ram_re),
        .ram_addr(ram_addr),
        .ram_dout0(ram_dout0),
        .ram_dout1(ram_dout1),
        .ram_dout2(ram_dout2),
        .ram_dout3(ram_dout3),
        // All 7x7 position outputs
        .patch_pe00_out(patch_pe00_out), .patch_pe01_out(patch_pe01_out), .patch_pe02_out(patch_pe02_out), .patch_pe03_out(patch_pe03_out), .patch_pe04_out(patch_pe04_out), .patch_pe05_out(patch_pe05_out), .patch_pe06_out(patch_pe06_out),
        .patch_pe10_out(patch_pe10_out), .patch_pe11_out(patch_pe11_out), .patch_pe12_out(patch_pe12_out), .patch_pe13_out(patch_pe13_out), .patch_pe14_out(patch_pe14_out), .patch_pe15_out(patch_pe15_out), .patch_pe16_out(patch_pe16_out),
        .patch_pe20_out(patch_pe20_out), .patch_pe21_out(patch_pe21_out), .patch_pe22_out(patch_pe22_out), .patch_pe23_out(patch_pe23_out), .patch_pe24_out(patch_pe24_out), .patch_pe25_out(patch_pe25_out), .patch_pe26_out(patch_pe26_out),
        .patch_pe30_out(patch_pe30_out), .patch_pe31_out(patch_pe31_out), .patch_pe32_out(patch_pe32_out), .patch_pe33_out(patch_pe33_out), .patch_pe34_out(patch_pe34_out), .patch_pe35_out(patch_pe35_out), .patch_pe36_out(patch_pe36_out),
        .patch_pe40_out(patch_pe40_out), .patch_pe41_out(patch_pe41_out), .patch_pe42_out(patch_pe42_out), .patch_pe43_out(patch_pe43_out), .patch_pe44_out(patch_pe44_out), .patch_pe45_out(patch_pe45_out), .patch_pe46_out(patch_pe46_out),
        .patch_pe50_out(patch_pe50_out), .patch_pe51_out(patch_pe51_out), .patch_pe52_out(patch_pe52_out), .patch_pe53_out(patch_pe53_out), .patch_pe54_out(patch_pe54_out), .patch_pe55_out(patch_pe55_out), .patch_pe56_out(patch_pe56_out),
        .patch_pe60_out(patch_pe60_out), .patch_pe61_out(patch_pe61_out), .patch_pe62_out(patch_pe62_out), .patch_pe63_out(patch_pe63_out), .patch_pe64_out(patch_pe64_out), .patch_pe65_out(patch_pe65_out), .patch_pe66_out(patch_pe66_out),
        .patches_valid(patches_valid)
    );

    // Instantiate Spatial Data Formatter
    spatial_data_formatter u_spatial_formatter (
        .clk(clk),
        .reset(reset),
        .start_formatting(start_formatting),
        .patches_valid(patches_valid),
        // All 7x7 position inputs
        .patch_pe00_in(patch_pe00_out), .patch_pe01_in(patch_pe01_out), .patch_pe02_in(patch_pe02_out), .patch_pe03_in(patch_pe03_out), .patch_pe04_in(patch_pe04_out), .patch_pe05_in(patch_pe05_out), .patch_pe06_in(patch_pe06_out),
        .patch_pe10_in(patch_pe10_out), .patch_pe11_in(patch_pe11_out), .patch_pe12_in(patch_pe12_out), .patch_pe13_in(patch_pe13_out), .patch_pe14_in(patch_pe14_out), .patch_pe15_in(patch_pe15_out), .patch_pe16_in(patch_pe16_out),
        .patch_pe20_in(patch_pe20_out), .patch_pe21_in(patch_pe21_out), .patch_pe22_in(patch_pe22_out), .patch_pe23_in(patch_pe23_out), .patch_pe24_in(patch_pe24_out), .patch_pe25_in(patch_pe25_out), .patch_pe26_in(patch_pe26_out),
        .patch_pe30_in(patch_pe30_out), .patch_pe31_in(patch_pe31_out), .patch_pe32_in(patch_pe32_out), .patch_pe33_in(patch_pe33_out), .patch_pe34_in(patch_pe34_out), .patch_pe35_in(patch_pe35_out), .patch_pe36_in(patch_pe36_out),
        .patch_pe40_in(patch_pe40_out), .patch_pe41_in(patch_pe41_out), .patch_pe42_in(patch_pe42_out), .patch_pe43_in(patch_pe43_out), .patch_pe44_in(patch_pe44_out), .patch_pe45_in(patch_pe45_out), .patch_pe46_in(patch_pe46_out),
        .patch_pe50_in(patch_pe50_out), .patch_pe51_in(patch_pe51_out), .patch_pe52_in(patch_pe52_out), .patch_pe53_in(patch_pe53_out), .patch_pe54_in(patch_pe54_out), .patch_pe55_in(patch_pe55_out), .patch_pe56_in(patch_pe56_out),
        .patch_pe60_in(patch_pe60_out), .patch_pe61_in(patch_pe61_out), .patch_pe62_in(patch_pe62_out), .patch_pe63_in(patch_pe63_out), .patch_pe64_in(patch_pe64_out), .patch_pe65_in(patch_pe65_out), .patch_pe66_in(patch_pe66_out),
        .formatted_A0(formatted_A0),
        .formatted_A1(formatted_A1),
        .formatted_A2(formatted_A2),
        .formatted_A3(formatted_A3),
        .formatted_data_valid(formatted_data_valid),
        .all_cols_sent(all_cols_sent),
        .next_block(next_block)
    );

endmodule 
