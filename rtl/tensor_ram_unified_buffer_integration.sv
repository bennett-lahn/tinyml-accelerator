// `include "sys_types.svh"

// module tensor_ram_unified_buffer_integration #(
//     parameter MAX_IMG_W = 64,
//     parameter MAX_IMG_H = 64,
//     parameter BUFFER_SIZE = 7,
//     parameter PATCH_SIZE = 4,
//     parameter PATCHES_PER_BLOCK = 4,
//     parameter MAX_CHANNELS = 64,
//     parameter MAX_PADDING = 3,
//     // Tensor RAM parameters
//     parameter READ_WIDTH = 128,
//     parameter DEPTH_WORDS = 1024,
//     parameter WRITE_WIDTH = 8,
//     parameter INIT_FILE = "../rtl/image_data.hex"
// )(
//     input logic clk,
//     input logic reset,
    
//     // Test control signals
//     input logic start_extraction,
//     input logic next_channel_group,
//     input logic next_spatial_block,
//     input logic start_formatting,
    
//     // Configuration
//     input logic [$clog2(MAX_IMG_W+1)-1:0] img_width,
//     input logic [$clog2(MAX_IMG_H+1)-1:0] img_height,
//     input logic [$clog2(MAX_CHANNELS+1)-1:0] num_channels,
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_top,
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_bottom,
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_left,
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_right,
    
//     // Tensor RAM write interface for test data loading
//     input logic tensor_ram_we,
//     input logic [$clog2(DEPTH_WORDS*(READ_WIDTH/WRITE_WIDTH))-1:0] tensor_ram_addr_w,
//     input logic [WRITE_WIDTH-1:0] tensor_ram_din,
    
//     // Status outputs
//     output logic block_ready,
//     output logic extraction_complete,
//     output logic all_channels_done,
//     output logic buffer_loading_complete,
//     output logic formatted_data_valid,
//     output logic all_cols_sent,
//     output logic next_block,
    
//     // Formatted data outputs (broken down for easier testing)
//     output int8_t formatted_A0_0, formatted_A0_1, formatted_A0_2, formatted_A0_3,
//     output int8_t formatted_A1_0, formatted_A1_1, formatted_A1_2, formatted_A1_3,
//     output int8_t formatted_A2_0, formatted_A2_1, formatted_A2_2, formatted_A2_3,
//     output int8_t formatted_A3_0, formatted_A3_1, formatted_A3_2, formatted_A3_3,
    
//     // Debug outputs: raw patch data from unified buffer
//     output logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out, 
//     output logic [31:0] patch_pe04_out, patch_pe05_out, patch_pe06_out,
//     output logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out,
//     output logic [31:0] patch_pe14_out, patch_pe15_out, patch_pe16_out,
//     output logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out,
//     output logic [31:0] patch_pe24_out, patch_pe25_out, patch_pe26_out,
//     output logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out,
//     output logic [31:0] patch_pe34_out, patch_pe35_out, patch_pe36_out,
//     output logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out,
//     output logic [31:0] patch_pe44_out, patch_pe45_out, patch_pe46_out,
//     output logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out,
//     output logic [31:0] patch_pe54_out, patch_pe55_out, patch_pe56_out,
//     output logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out,
//     output logic [31:0] patch_pe64_out, patch_pe65_out, patch_pe66_out,
//     output logic patches_valid
// );

//     // Internal signals
//     logic ram_re;
//     logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr_full;
//     logic [$clog2(DEPTH_WORDS)-1:0] ram_addr_truncated;
//     logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3;
//     logic [READ_WIDTH-1:0] ram_dout;
    
//     // Formatted data arrays
//     int8_t formatted_A0 [0:3];
//     int8_t formatted_A1 [0:3];
//     int8_t formatted_A2 [0:3];
//     int8_t formatted_A3 [0:3];
    
//     // Assign array elements to individual outputs for easier testing
//     assign formatted_A0_0 = formatted_A0[0];
//     assign formatted_A0_1 = formatted_A0[1];
//     assign formatted_A0_2 = formatted_A0[2];
//     assign formatted_A0_3 = formatted_A0[3];
    
//     assign formatted_A1_0 = formatted_A1[0];
//     assign formatted_A1_1 = formatted_A1[1];
//     assign formatted_A1_2 = formatted_A1[2];
//     assign formatted_A1_3 = formatted_A1[3];
    
//     assign formatted_A2_0 = formatted_A2[0];
//     assign formatted_A2_1 = formatted_A2[1];
//     assign formatted_A2_2 = formatted_A2[2];
//     assign formatted_A2_3 = formatted_A2[3];
    
//     assign formatted_A3_0 = formatted_A3[0];
//     assign formatted_A3_1 = formatted_A3[1];
//     assign formatted_A3_2 = formatted_A3[2];
//     assign formatted_A3_3 = formatted_A3[3];

//     // Truncate address to fit tensor RAM requirements
//     assign ram_addr_truncated = ram_addr_full[$clog2(DEPTH_WORDS)-1:0];

//     // Tensor RAM instance
//     tensor_ram #(
//         .READ_WIDTH(READ_WIDTH),
//         .DEPTH_WORDS(DEPTH_WORDS),
//         .WRITE_WIDTH(WRITE_WIDTH),
//         .INIT_FILE(INIT_FILE)
//     ) tensor_ram_inst (
//         .clk(clk),
//         .we(tensor_ram_we),
//         .re(ram_re),
//         .addr_w(tensor_ram_addr_w),
//         .din(tensor_ram_din),
//         .addr_r(ram_addr_truncated),
//         .dout(ram_dout),
//         .dout0(ram_dout0),
//         .dout1(ram_dout1),
//         .dout2(ram_dout2),
//         .dout3(ram_dout3)
//     );

//     // Unified Buffer instance
//     unified_buffer #(
//         .MAX_IMG_W(MAX_IMG_W),
//         .MAX_IMG_H(MAX_IMG_H),
//         .BUFFER_SIZE(BUFFER_SIZE),
//         .PATCH_SIZE(PATCH_SIZE),
//         .PATCHES_PER_BLOCK(PATCHES_PER_BLOCK),
//         .MAX_CHANNELS(MAX_CHANNELS),
//         .MAX_PADDING(MAX_PADDING)
//     ) unified_buffer_inst (
//         .clk(clk),
//         .reset(reset),
//         .start_extraction(start_extraction),
//         .next_channel_group(next_channel_group),
//         .next_spatial_block(next_spatial_block),
//         .img_width(img_width),
//         .img_height(img_height),
//         .num_channels(num_channels),
//         .pad_top(pad_top),
//         .pad_bottom(pad_bottom),
//         .pad_left(pad_left),
//         .pad_right(pad_right),
//         .ram_re(ram_re),
//         .ram_addr(ram_addr_full),
//         .ram_dout0(ram_dout0),
//         .ram_dout1(ram_dout1),
//         .ram_dout2(ram_dout2),
//         .ram_dout3(ram_dout3),
//         .block_ready(block_ready),
//         .extraction_complete(extraction_complete),
//         .all_channels_done(all_channels_done),
//         .buffer_loading_complete(buffer_loading_complete),
//         .patch_pe00_out(patch_pe00_out),
//         .patch_pe01_out(patch_pe01_out),
//         .patch_pe02_out(patch_pe02_out),
//         .patch_pe03_out(patch_pe03_out),
//         .patch_pe04_out(patch_pe04_out),
//         .patch_pe05_out(patch_pe05_out),
//         .patch_pe06_out(patch_pe06_out),
//         .patch_pe10_out(patch_pe10_out),
//         .patch_pe11_out(patch_pe11_out),
//         .patch_pe12_out(patch_pe12_out),
//         .patch_pe13_out(patch_pe13_out),
//         .patch_pe14_out(patch_pe14_out),
//         .patch_pe15_out(patch_pe15_out),
//         .patch_pe16_out(patch_pe16_out),
//         .patch_pe20_out(patch_pe20_out),
//         .patch_pe21_out(patch_pe21_out),
//         .patch_pe22_out(patch_pe22_out),
//         .patch_pe23_out(patch_pe23_out),
//         .patch_pe24_out(patch_pe24_out),
//         .patch_pe25_out(patch_pe25_out),
//         .patch_pe26_out(patch_pe26_out),
//         .patch_pe30_out(patch_pe30_out),
//         .patch_pe31_out(patch_pe31_out),
//         .patch_pe32_out(patch_pe32_out),
//         .patch_pe33_out(patch_pe33_out),
//         .patch_pe34_out(patch_pe34_out),
//         .patch_pe35_out(patch_pe35_out),
//         .patch_pe36_out(patch_pe36_out),
//         .patch_pe40_out(patch_pe40_out),
//         .patch_pe41_out(patch_pe41_out),
//         .patch_pe42_out(patch_pe42_out),
//         .patch_pe43_out(patch_pe43_out),
//         .patch_pe44_out(patch_pe44_out),
//         .patch_pe45_out(patch_pe45_out),
//         .patch_pe46_out(patch_pe46_out),
//         .patch_pe50_out(patch_pe50_out),
//         .patch_pe51_out(patch_pe51_out),
//         .patch_pe52_out(patch_pe52_out),
//         .patch_pe53_out(patch_pe53_out),
//         .patch_pe54_out(patch_pe54_out),
//         .patch_pe55_out(patch_pe55_out),
//         .patch_pe56_out(patch_pe56_out),
//         .patch_pe60_out(patch_pe60_out),
//         .patch_pe61_out(patch_pe61_out),
//         .patch_pe62_out(patch_pe62_out),
//         .patch_pe63_out(patch_pe63_out),
//         .patch_pe64_out(patch_pe64_out),
//         .patch_pe65_out(patch_pe65_out),
//         .patch_pe66_out(patch_pe66_out),
//         .patches_valid(patches_valid)
//     );

//     // Spatial Data Formatter instance
//     spatial_data_formatter spatial_data_formatter_inst (
//         .clk(clk),
//         .reset(reset),
//         .start_formatting(start_formatting),
//         .patches_valid(patches_valid),
//         .patch_pe00_in(patch_pe00_out),
//         .patch_pe01_in(patch_pe01_out),
//         .patch_pe02_in(patch_pe02_out),
//         .patch_pe03_in(patch_pe03_out),
//         .patch_pe04_in(patch_pe04_out),
//         .patch_pe05_in(patch_pe05_out),
//         .patch_pe06_in(patch_pe06_out),
//         .patch_pe10_in(patch_pe10_out),
//         .patch_pe11_in(patch_pe11_out),
//         .patch_pe12_in(patch_pe12_out),
//         .patch_pe13_in(patch_pe13_out),
//         .patch_pe14_in(patch_pe14_out),
//         .patch_pe15_in(patch_pe15_out),
//         .patch_pe16_in(patch_pe16_out),
//         .patch_pe20_in(patch_pe20_out),
//         .patch_pe21_in(patch_pe21_out),
//         .patch_pe22_in(patch_pe22_out),
//         .patch_pe23_in(patch_pe23_out),
//         .patch_pe24_in(patch_pe24_out),
//         .patch_pe25_in(patch_pe25_out),
//         .patch_pe26_in(patch_pe26_out),
//         .patch_pe30_in(patch_pe30_out),
//         .patch_pe31_in(patch_pe31_out),
//         .patch_pe32_in(patch_pe32_out),
//         .patch_pe33_in(patch_pe33_out),
//         .patch_pe34_in(patch_pe34_out),
//         .patch_pe35_in(patch_pe35_out),
//         .patch_pe36_in(patch_pe36_out),
//         .patch_pe40_in(patch_pe40_out),
//         .patch_pe41_in(patch_pe41_out),
//         .patch_pe42_in(patch_pe42_out),
//         .patch_pe43_in(patch_pe43_out),
//         .patch_pe44_in(patch_pe44_out),
//         .patch_pe45_in(patch_pe45_out),
//         .patch_pe46_in(patch_pe46_out),
//         .patch_pe50_in(patch_pe50_out),
//         .patch_pe51_in(patch_pe51_out),
//         .patch_pe52_in(patch_pe52_out),
//         .patch_pe53_in(patch_pe53_out),
//         .patch_pe54_in(patch_pe54_out),
//         .patch_pe55_in(patch_pe55_out),
//         .patch_pe56_in(patch_pe56_out),
//         .patch_pe60_in(patch_pe60_out),
//         .patch_pe61_in(patch_pe61_out),
//         .patch_pe62_in(patch_pe62_out),
//         .patch_pe63_in(patch_pe63_out),
//         .patch_pe64_in(patch_pe64_out),
//         .patch_pe65_in(patch_pe65_out),
//         .patch_pe66_in(patch_pe66_out),
//         .formatted_A0(formatted_A0),
//         .formatted_A1(formatted_A1),
//         .formatted_A2(formatted_A2),
//         .formatted_A3(formatted_A3),
//         .formatted_data_valid(formatted_data_valid),
//         .all_cols_sent(all_cols_sent),
//         .next_block(next_block)
//     );

// endmodule 

