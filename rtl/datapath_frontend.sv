`include "sys_types.svh"
module datapath_frontend
#
(
    parameter int DEPTH_128B_WORDS = 128,
    parameter int ADDR_BITS = $clog2(DEPTH_128B_WORDS),
    parameter int MAX_N = 64,
    parameter int N_BITS = $clog2(MAX_N),
    parameter int MAX_NUM_CH = 64,
    parameter int CH_BITS = $clog2(MAX_NUM_CH+1),
    parameter string INIT_FILE = "../rtl/image_data.hex"
    ,parameter int MAX_IMG_W = 64
    ,parameter int MAX_IMG_H = 64
    ,parameter int MAX_PADDING = 3
)
(
    input logic clk
    ,input logic reset
    ,input logic [2:0] layer_idx
    ,input logic start_extraction
    ,input logic next_channel_group
    ,input logic next_spatial_block
    ,input logic start_formatting
);

    //tensor ram signals 
    logic tensor_ram_we;
    logic tensor_ram_re;
    logic [N_BITS-1:0] tensor_ram_write_row;
    logic [N_BITS-1:0] tensor_ram_write_col;
    logic [CH_BITS-1:0] tensor_ram_write_channel;
    logic [N_BITS-1:0] tensor_ram_num_cols;
    logic [CH_BITS-1:0] tensor_ram_num_channels; 
    logic [7:0] tensor_ram_data_in;
    logic [ADDR_BITS-1:0] tensor_ram_read_addr;
    logic [31:0] tensor_ram_dout0;
    logic [31:0] tensor_ram_dout1;
    logic [31:0] tensor_ram_dout2;
    logic [31:0] tensor_ram_dout3;
    logic ram_data_valid;



    tensor_ram #(
        .DEPTH_128B_WORDS(DEPTH_128B_WORDS),
        .ADDR_BITS(ADDR_BITS),
        .MAX_N(MAX_N),
        .N_BITS(N_BITS),
        .MAX_NUM_CH(MAX_NUM_CH),
        .CH_BITS(CH_BITS),
        .INIT_FILE(INIT_FILE)
    ) tensor_ram_inst (
        .clk(clk)
        ,.reset(reset)
        ,.write_en(tensor_ram_we)
        ,.read_en(tensor_ram_re)
        ,.write_row(tensor_ram_write_row)
        ,.write_col(tensor_ram_write_col)
        ,.write_channel(tensor_ram_write_channel)
        ,.num_cols(tensor_ram_num_cols)
        ,.num_channels(tensor_ram_num_channels)
        ,.data_in(tensor_ram_data_in)
        ,.read_addr(tensor_ram_read_addr)
        ,.ram_dout0(tensor_ram_dout0)
        ,.ram_dout1(tensor_ram_dout1)
        ,.ram_dout2(tensor_ram_dout2)
        ,.ram_dout3(tensor_ram_dout3)
        ,.data_valid(ram_data_valid)
    );

    //unified buffer signals
    // logic layer_idx;
    // logic start_extraction;
    // logic next_channel_group;
    // logic next_spatial_block;

    logic ram_re;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_NUM_CH/4)-1:0] ram_addr;
    logic [31:0] ram_dout0;
    logic [31:0] ram_dout1;
    logic [31:0] ram_dout2;
    logic [31:0] ram_dout3;


    logic block_ready;
    logic extraction_complete;
    logic all_channels_done;
    logic buffer_loading_complete;

    logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr;
    logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr;
    logic block_coords_valid;

    //patch outputs
    logic [31:0] patch_pe00_out;
    logic [31:0] patch_pe01_out;
    logic [31:0] patch_pe02_out;
    logic [31:0] patch_pe03_out;
    logic [31:0] patch_pe04_out;
    logic [31:0] patch_pe05_out;
    logic [31:0] patch_pe06_out;
    logic [31:0] patch_pe10_out;
    logic [31:0] patch_pe11_out;
    logic [31:0] patch_pe12_out;
    logic [31:0] patch_pe13_out;
    logic [31:0] patch_pe14_out;
    logic [31:0] patch_pe15_out;
    logic [31:0] patch_pe16_out;
    logic [31:0] patch_pe20_out;
    logic [31:0] patch_pe21_out;
    logic [31:0] patch_pe22_out;
    logic [31:0] patch_pe23_out;
    logic [31:0] patch_pe24_out;
    logic [31:0] patch_pe25_out;
    logic [31:0] patch_pe26_out;
    logic [31:0] patch_pe30_out;
    logic [31:0] patch_pe31_out;
    logic [31:0] patch_pe32_out;
    logic [31:0] patch_pe33_out;
    logic [31:0] patch_pe34_out;
    logic [31:0] patch_pe35_out;
    logic [31:0] patch_pe36_out;
    logic [31:0] patch_pe40_out;
    logic [31:0] patch_pe41_out;
    logic [31:0] patch_pe42_out;
    logic [31:0] patch_pe43_out;    
    logic [31:0] patch_pe44_out;
    logic [31:0] patch_pe45_out;
    logic [31:0] patch_pe46_out;
    logic [31:0] patch_pe50_out;
    logic [31:0] patch_pe51_out;
    logic [31:0] patch_pe52_out;
    logic [31:0] patch_pe53_out;
    logic [31:0] patch_pe54_out;
    logic [31:0] patch_pe55_out;
    logic [31:0] patch_pe56_out;
    logic [31:0] patch_pe60_out;
    logic [31:0] patch_pe61_out;
    logic [31:0] patch_pe62_out;
    logic [31:0] patch_pe63_out;
    logic [31:0] patch_pe64_out;
    logic [31:0] patch_pe65_out;
    logic [31:0] patch_pe66_out;
    logic patches_valid;

    
    unified_buffer_harness #(
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MAX_CHANNELS(MAX_NUM_CH),
        .MAX_PADDING(MAX_PADDING)
    ) 
    unified_buffer_harness_inst
    (
        .clk(clk),
        .reset(reset),
        .current_layer_idx(layer_idx),
        .start_extraction(start_extraction),
        .next_channel_group(next_channel_group),
        .next_spatial_block(next_spatial_block),
        .ram_re(ram_re),
        .ram_addr(ram_addr),
        .ram_dout0(ram_dout0),
        .ram_dout1(ram_dout1),
        .ram_dout2(ram_dout2),
        .ram_dout3(ram_dout3),
        .ram_data_valid(ram_data_valid),
        .block_ready(block_ready),
        .extraction_complete(extraction_complete),
        .all_channels_done(all_channels_done),
        .buffer_loading_complete(buffer_loading_complete),
        .block_start_col_addr(block_start_col_addr),
        .block_start_row_addr(block_start_row_addr),
        .block_coords_valid(block_coords_valid),
        .patch_pe00_out(patch_pe00_out),
        .patch_pe01_out(patch_pe01_out),
        .patch_pe02_out(patch_pe02_out),
        .patch_pe03_out(patch_pe03_out),
        .patch_pe04_out(patch_pe04_out),
        .patch_pe05_out(patch_pe05_out),
        .patch_pe06_out(patch_pe06_out),
        .patch_pe10_out(patch_pe10_out),
        .patch_pe11_out(patch_pe11_out),
        .patch_pe12_out(patch_pe12_out),
        .patch_pe13_out(patch_pe13_out),
        .patch_pe14_out(patch_pe14_out),
        .patch_pe15_out(patch_pe15_out),
        .patch_pe16_out(patch_pe16_out),
        .patch_pe20_out(patch_pe20_out),
        .patch_pe21_out(patch_pe21_out),
        .patch_pe22_out(patch_pe22_out),
        .patch_pe23_out(patch_pe23_out),
        .patch_pe24_out(patch_pe24_out),
        .patch_pe25_out(patch_pe25_out),
        .patch_pe26_out(patch_pe26_out),
        .patch_pe30_out(patch_pe30_out),
        .patch_pe31_out(patch_pe31_out),
        .patch_pe32_out(patch_pe32_out),
        .patch_pe33_out(patch_pe33_out),
        .patch_pe34_out(patch_pe34_out),
        .patch_pe35_out(patch_pe35_out),
        .patch_pe36_out(patch_pe36_out),
        .patch_pe40_out(patch_pe40_out),
        .patch_pe41_out(patch_pe41_out),
        .patch_pe42_out(patch_pe42_out),
        .patch_pe43_out(patch_pe43_out),
        .patch_pe44_out(patch_pe44_out),
        .patch_pe45_out(patch_pe45_out),
        .patch_pe46_out(patch_pe46_out),
        .patch_pe50_out(patch_pe50_out),
        .patch_pe51_out(patch_pe51_out),
        .patch_pe52_out(patch_pe52_out),
        .patch_pe53_out(patch_pe53_out),
        .patch_pe54_out(patch_pe54_out),
        .patch_pe55_out(patch_pe55_out),
        .patch_pe56_out(patch_pe56_out),
        .patch_pe60_out(patch_pe60_out),
        .patch_pe61_out(patch_pe61_out),
        .patch_pe62_out(patch_pe62_out),
        .patch_pe63_out(patch_pe63_out),
        .patch_pe64_out(patch_pe64_out),
        .patch_pe65_out(patch_pe65_out),
        .patch_pe66_out(patch_pe66_out),
        .patches_valid(patches_valid)
    );
    



    //spatial data formatter
    logic all_cols_sent;
    logic next_block;
    int8_t formatted_A0 [0:3];
    int8_t formatted_A1 [0:3];
    int8_t formatted_A2 [0:3];
    int8_t formatted_A3 [0:3];
    logic formatted_data_valid;
    

    spatial_data_formatter spatial_data_formatter_inst (
        .clk(clk),
        .reset(reset),
        .start_formatting(start_formatting),
        .patches_valid(patches_valid),
        .patch_pe00_in(patch_pe00_out),
        .patch_pe01_in(patch_pe01_out),
        .patch_pe02_in(patch_pe02_out),
        .patch_pe03_in(patch_pe03_out),
        .patch_pe04_in(patch_pe04_out),
        .patch_pe05_in(patch_pe05_out),
        .patch_pe06_in(patch_pe06_out),
        .patch_pe10_in(patch_pe10_out),
        .patch_pe11_in(patch_pe11_out),
        .patch_pe12_in(patch_pe12_out),
        .patch_pe13_in(patch_pe13_out),
        .patch_pe14_in(patch_pe14_out),
        .patch_pe15_in(patch_pe15_out),
        .patch_pe16_in(patch_pe16_out),
        .patch_pe20_in(patch_pe20_out),
        .patch_pe21_in(patch_pe21_out),
        .patch_pe22_in(patch_pe22_out),
        .patch_pe23_in(patch_pe23_out),
        .patch_pe24_in(patch_pe24_out),
        .patch_pe25_in(patch_pe25_out),
        .patch_pe26_in(patch_pe26_out),
        .patch_pe30_in(patch_pe30_out),
        .patch_pe31_in(patch_pe31_out),
        .patch_pe32_in(patch_pe32_out),
        .patch_pe33_in(patch_pe33_out),
        .patch_pe34_in(patch_pe34_out),
        .patch_pe35_in(patch_pe35_out),
        .patch_pe36_in(patch_pe36_out),
        .patch_pe40_in(patch_pe40_out),
        .patch_pe41_in(patch_pe41_out),
        .patch_pe42_in(patch_pe42_out),
        .patch_pe43_in(patch_pe43_out),
        .patch_pe44_in(patch_pe44_out),
        .patch_pe45_in(patch_pe45_out),
        .patch_pe46_in(patch_pe46_out),
        .patch_pe50_in(patch_pe50_out),
        .patch_pe51_in(patch_pe51_out),
        .patch_pe52_in(patch_pe52_out),
        .patch_pe53_in(patch_pe53_out),
        .patch_pe54_in(patch_pe54_out),
        .patch_pe55_in(patch_pe55_out),
        .patch_pe56_in(patch_pe56_out),
        .patch_pe60_in(patch_pe60_out),
        .patch_pe61_in(patch_pe61_out),
        .patch_pe62_in(patch_pe62_out),
        .patch_pe63_in(patch_pe63_out),
        .patch_pe64_in(patch_pe64_out),
        .patch_pe65_in(patch_pe65_out),
        .patch_pe66_in(patch_pe66_out),
        .formatted_A0(formatted_A0),
        .formatted_A1(formatted_A1),
        .formatted_A2(formatted_A2),
        .formatted_A3(formatted_A3),
        .formatted_data_valid(formatted_data_valid),
        .all_cols_sent(all_cols_sent),
        .next_block(next_block)
    );




    assign tensor_ram_we = 1'b0;
    assign tensor_ram_write_row = 'b0;
    assign tensor_ram_write_col = 'b0;
    assign tensor_ram_write_channel = 'b0;
    assign tensor_ram_num_cols = 'b0;
    assign tensor_ram_num_channels = 'b0;
    assign tensor_ram_data_in = '0;
    /* verilator lint_off WIDTHTRUNC */
    assign tensor_ram_read_addr = ram_addr;
    /* verilator lint_on WIDTHTRUNC */
    assign tensor_ram_re = ram_re;
    assign ram_dout0 = tensor_ram_dout0;
    assign ram_dout1 = tensor_ram_dout1;
    assign ram_dout2 = tensor_ram_dout2;
    assign ram_dout3 = tensor_ram_dout3;





endmodule  
