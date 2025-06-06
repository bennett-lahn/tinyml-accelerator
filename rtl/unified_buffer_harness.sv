// Example integration of layer_specific_extractor into TPU datapath
// This shows how to replace the unified_buffer with the layer-specific version

module unified_buffer_harness #(
    parameter MAX_IMG_W = 64,
    parameter MAX_IMG_H = 64,
    parameter MAX_CHANNELS = 64,
    parameter MAX_PADDING = 3
)(
    input logic clk,
    input logic reset,
    
    // Layer control - this would come from your main controller will be used to mux between the different unif_bufs
    input logic [2:0] current_layer_idx,  // Updated by controller as layers progress
    
    // Control signals (same interface as unified_buffer)
    input logic start_extraction,
    input logic next_channel_group,
    input logic next_spatial_block,
    
    // Memory interface (same as unified_buffer)
    output logic ram_re,
    output logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr,
    input logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3,
    input logic ram_data_valid,
    
    // Status outputs (same as unified_buffer)
    output logic all_channels_done,
    output logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr,
    output logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr,
    
    // Patch outputs - mapped to existing PE grid interface
    // Map variable patch sizes to fixed 7x7 PE grid for compatibility
    output logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out,
    output logic [31:0] patch_pe04_out, patch_pe05_out, patch_pe06_out,
    output logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out,
    output logic [31:0] patch_pe14_out, patch_pe15_out, patch_pe16_out,
    output logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out,
    output logic [31:0] patch_pe24_out, patch_pe25_out, patch_pe26_out,
    output logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out,
    output logic [31:0] patch_pe34_out, patch_pe35_out, patch_pe36_out,
    output logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out,
    output logic [31:0] patch_pe44_out, patch_pe45_out, patch_pe46_out,
    output logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out,
    output logic [31:0] patch_pe54_out, patch_pe55_out, patch_pe56_out,
    output logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out,
    output logic [31:0] patch_pe64_out, patch_pe65_out, patch_pe66_out,
    output logic patches_valid
);

    logic start_extraction_0;
    logic next_channel_group_0;
    logic next_spatial_block_0;
    // Internal patch data from layer-specific extractor
    logic [31:0] layer_patch_data_0 [6:0][6:0];
    logic layer_patches_valid_0;

    logic ram_re_0;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr_0;
    logic [31:0] ram_dout0_0, ram_dout1_0, ram_dout2_0, ram_dout3_0;
    logic ram_data_valid_0;

    logic all_channels_done_0;
    logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr_0;
    logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr_0;

    logic [31:0] layer_patch_data [6:0][6:0];

    // Instantiate the layer-specific extractor
    unif_buf_0 #(
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MAX_CHANNELS(MAX_CHANNELS),
        .MAX_PADDING(MAX_PADDING)
    ) unif_buf_0_inst (
        .clk(clk),
        .reset(reset),
        .start_extraction(start_extraction_0),
        .next_channel_group(next_channel_group_0),
        .next_spatial_block(next_spatial_block_0),
        .ram_re(ram_re_0),
        .ram_addr(ram_addr_0),
        .ram_dout0(ram_dout0_0),
        .ram_dout1(ram_dout1_0),
        .ram_dout2(ram_dout2_0),
        .ram_dout3(ram_dout3_0),
        .ram_data_valid(ram_data_valid_0),
        .all_channels_done(all_channels_done_0),
        .block_start_col_addr(block_start_col_addr_0),
        .block_start_row_addr(block_start_row_addr_0),
        .patch_data_out(layer_patch_data_0),
        .patches_valid(layer_patches_valid_0)
    );

    logic start_extraction_1;
    logic next_channel_group_1;
    logic next_spatial_block_1;
    logic ram_re_1;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr_1;
    logic [31:0] ram_dout0_1, ram_dout1_1, ram_dout2_1, ram_dout3_1;
    logic ram_data_valid_1;
    logic all_channels_done_1;
    logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr_1;
    logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr_1;
    logic [31:0] layer_patch_data_1 [6:0][6:0];
    logic layer_patches_valid_1;

    unif_buf_1 #(
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MAX_CHANNELS(MAX_CHANNELS),
        .MAX_PADDING(MAX_PADDING)
    ) unif_buf_1_inst (
        .clk(clk),
        .reset(reset),
        .start_extraction(start_extraction_1),
        .next_channel_group(next_channel_group_1),
        .next_spatial_block(next_spatial_block_1),
        .ram_re(ram_re_1),
        .ram_addr(ram_addr_1),
        .ram_dout0(ram_dout0_1),
        .ram_dout1(ram_dout1_1),
        .ram_dout2(ram_dout2_1),
        .ram_dout3(ram_dout3_1),
        .ram_data_valid(ram_data_valid_1),
        .all_channels_done(all_channels_done_1),
        .block_start_col_addr(block_start_col_addr_1),
        .block_start_row_addr(block_start_row_addr_1),
        .patch_data_out(layer_patch_data_1),
        .patches_valid(layer_patches_valid_1)
    );

    logic start_extraction_2;
    logic next_channel_group_2;
    logic next_spatial_block_2;
    logic ram_re_2;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr_2;
    logic [31:0] ram_dout0_2, ram_dout1_2, ram_dout2_2, ram_dout3_2;
    logic ram_data_valid_2;
    logic all_channels_done_2;
    logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr_2;
    logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr_2;
    logic [31:0] layer_patch_data_2 [6:0][6:0];
    logic layer_patches_valid_2;

    unif_buf_2 #(
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MAX_CHANNELS(MAX_CHANNELS),
        .MAX_PADDING(MAX_PADDING)
    ) unif_buf_2_inst (
        .clk(clk),
        .reset(reset),
        .start_extraction(start_extraction_2),
        .next_channel_group(next_channel_group_2),
        .next_spatial_block(next_spatial_block_2),
        .ram_re(ram_re_2),
        .ram_addr(ram_addr_2),
        .ram_dout0(ram_dout0_2),
        .ram_dout1(ram_dout1_2),
        .ram_dout2(ram_dout2_2),
        .ram_dout3(ram_dout3_2),
        .ram_data_valid(ram_data_valid_2),
        .all_channels_done(all_channels_done_2),
        .block_start_col_addr(block_start_col_addr_2),
        .block_start_row_addr(block_start_row_addr_2),
        .patch_data_out(layer_patch_data_2),
        .patches_valid(layer_patches_valid_2)
    );

    logic start_extraction_3;
    logic next_channel_group_3;
    logic next_spatial_block_3;
    logic ram_re_3;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr_3;
    logic [31:0] ram_dout0_3, ram_dout1_3, ram_dout2_3, ram_dout3_3;
    logic ram_data_valid_3;
    logic all_channels_done_3;
    logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr_3;
    logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr_3;
    logic [31:0] layer_patch_data_3 [6:0][6:0];
    logic layer_patches_valid_3;

    unif_buf_3 #(
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MAX_CHANNELS(MAX_CHANNELS),
        .MAX_PADDING(MAX_PADDING)
    ) unif_buf_3_inst (
        .clk(clk),
        .reset(reset),
        .start_extraction(start_extraction_3),
        .next_channel_group(next_channel_group_3),
        .next_spatial_block(next_spatial_block_3),
        .ram_re(ram_re_3),
        .ram_addr(ram_addr_3),
        .ram_dout0(ram_dout0_3),
        .ram_dout1(ram_dout1_3),
        .ram_dout2(ram_dout2_3),
        .ram_dout3(ram_dout3_3),
        .ram_data_valid(ram_data_valid_3),
        .all_channels_done(all_channels_done_3),
        .block_start_col_addr(block_start_col_addr_3),
        .block_start_row_addr(block_start_row_addr_3),
        .patch_data_out(layer_patch_data_3),
        .patches_valid(layer_patches_valid_3)
    );
    
    
    //mux between the different layer-specific buffers
    always_comb begin
        case (current_layer_idx)
            0: begin
                start_extraction_0 = start_extraction;
                next_channel_group_0 = next_channel_group;
                next_spatial_block_0 = next_spatial_block;
                ram_re = ram_re_0;
                ram_addr = ram_addr_0;
                ram_dout0_0 = ram_dout0;
                ram_dout1_0 = ram_dout1;
                ram_dout2_0 = ram_dout2;
                ram_dout3_0 = ram_dout3;
                ram_data_valid_0 = ram_data_valid;
                all_channels_done = all_channels_done_0;
                layer_patch_data = layer_patch_data_0;
                block_start_col_addr = block_start_col_addr_0;
                block_start_row_addr = block_start_row_addr_0;
                patches_valid = layer_patches_valid_0;
            end
            1: begin
                start_extraction_1 = start_extraction;
                next_channel_group_1 = next_channel_group;
                next_spatial_block_1 = next_spatial_block;
                ram_re = ram_re_1;
                ram_addr = ram_addr_1;
                ram_dout0_1 = ram_dout0;
                ram_dout1_1 = ram_dout1;
                ram_dout2_1 = ram_dout2;
                ram_dout3_1 = ram_dout3;
                ram_data_valid_1 = ram_data_valid;
                all_channels_done = all_channels_done_1;
                layer_patch_data = layer_patch_data_1;
                block_start_col_addr = block_start_col_addr_1;
                block_start_row_addr = block_start_row_addr_1;
                layer_patch_data = layer_patch_data_1;
                patches_valid = layer_patches_valid_1;
            end
            2: begin
                start_extraction_2 = start_extraction;
                next_channel_group_2 = next_channel_group;
                next_spatial_block_2 = next_spatial_block;
                ram_re = ram_re_2;
                ram_addr = ram_addr_2;
                ram_dout0_2 = ram_dout0;
                ram_dout1_2 = ram_dout1;
                ram_dout2_2 = ram_dout2;
                ram_dout3_2 = ram_dout3;
                ram_data_valid_2 = ram_data_valid;
                all_channels_done = all_channels_done_2;
                layer_patch_data = layer_patch_data_2;
                block_start_col_addr = block_start_col_addr_2;
                block_start_row_addr = block_start_row_addr_2;
                layer_patch_data = layer_patch_data_2;
                patches_valid = layer_patches_valid_2;
            end
            3: begin
                start_extraction_3 = start_extraction;
                next_channel_group_3 = next_channel_group;
                next_spatial_block_3 = next_spatial_block;
                ram_re = ram_re_3;
                ram_addr = ram_addr_3;
                ram_dout0_3 = ram_dout0;
                ram_dout1_3 = ram_dout1;
                ram_dout2_3 = ram_dout2;
                ram_dout3_3 = ram_dout3;
                ram_data_valid_3 = ram_data_valid;
                all_channels_done = all_channels_done_3;
                layer_patch_data = layer_patch_data_3;
                block_start_col_addr = block_start_col_addr_3;
                block_start_row_addr = block_start_row_addr_3;
                layer_patch_data = layer_patch_data_3;
                patches_valid = layer_patches_valid_3;
            end
        endcase
    end
        
    // Map variable-sized patches to fixed PE grid
    // This maintains compatibility with existing PE array expectations
    always_comb begin

        // Map layer-specific patch data based on layer configuration
        patch_pe00_out = layer_patch_data[0][0]; patch_pe01_out = layer_patch_data[0][1];
        patch_pe02_out = layer_patch_data[0][2]; patch_pe03_out = layer_patch_data[0][3];
        patch_pe04_out = layer_patch_data[0][4]; patch_pe05_out = layer_patch_data[0][5];
        patch_pe06_out = layer_patch_data[0][6];
        
        patch_pe10_out = layer_patch_data[1][0]; patch_pe11_out = layer_patch_data[1][1];
        patch_pe12_out = layer_patch_data[1][2]; patch_pe13_out = layer_patch_data[1][3];
        patch_pe14_out = layer_patch_data[1][4]; patch_pe15_out = layer_patch_data[1][5];
        patch_pe16_out = layer_patch_data[1][6];
        
        patch_pe20_out = layer_patch_data[2][0]; patch_pe21_out = layer_patch_data[2][1];
        patch_pe22_out = layer_patch_data[2][2]; patch_pe23_out = layer_patch_data[2][3];
        patch_pe24_out = layer_patch_data[2][4]; patch_pe25_out = layer_patch_data[2][5];
        patch_pe26_out = layer_patch_data[2][6];
        
        patch_pe30_out = layer_patch_data[3][0]; patch_pe31_out = layer_patch_data[3][1];
        patch_pe32_out = layer_patch_data[3][2]; patch_pe33_out = layer_patch_data[3][3];
        patch_pe34_out = layer_patch_data[3][4]; patch_pe35_out = layer_patch_data[3][5];
        patch_pe36_out = layer_patch_data[3][6];
        
        patch_pe40_out = layer_patch_data[4][0]; patch_pe41_out = layer_patch_data[4][1];
        patch_pe42_out = layer_patch_data[4][2]; patch_pe43_out = layer_patch_data[4][3];
        patch_pe44_out = layer_patch_data[4][4]; patch_pe45_out = layer_patch_data[4][5];
        patch_pe46_out = layer_patch_data[4][6];
        
        patch_pe50_out = layer_patch_data[5][0]; patch_pe51_out = layer_patch_data[5][1];
        patch_pe52_out = layer_patch_data[5][2]; patch_pe53_out = layer_patch_data[5][3];
        patch_pe54_out = layer_patch_data[5][4]; patch_pe55_out = layer_patch_data[5][5];
        patch_pe56_out = layer_patch_data[5][6];
        
        patch_pe60_out = layer_patch_data[6][0]; patch_pe61_out = layer_patch_data[6][1];
        patch_pe62_out = layer_patch_data[6][2]; patch_pe63_out = layer_patch_data[6][3];
        patch_pe64_out = layer_patch_data[6][4]; patch_pe65_out = layer_patch_data[6][5];
        patch_pe66_out = layer_patch_data[6][6];
    end

endmodule

// Usage example in main TPU datapath:
/*
    // Replace the existing unified_buffer instantiation with:
    
    layer_specific_integration_example #(
        .MAX_IMG_W(IMG_W),
        .MAX_IMG_H(IMG_H),
        .MAX_CHANNELS(MAX_NUM_CH),
        .MAX_PADDING(MAX_PADDING)
    ) LAYER_SPECIFIC_BUFFER (
        .clk(clk),
        .reset(reset | reset_datapath),
        .current_layer_idx(current_layer_index),  // New input from controller
        .start_extraction(start_block_extraction),
        .next_channel_group(next_channel_group),
        .next_spatial_block(next_spatial_block),
        .ram_re(buffer_ram_read_enable),
        .ram_addr(buffer_ram_addr),
        .ram_dout0(buffer_ram_in0),
        .ram_dout1(buffer_ram_in1),
        .ram_dout2(buffer_ram_in2),
        .ram_dout3(buffer_ram_in3),
        .all_channels_done(all_channels_done),
        .block_start_col_addr(block_start_col_addr),
        .block_start_row_addr(block_start_row_addr),
        // All the patch outputs map to existing signals...
        .patch_pe00_out(patch_pe00_out),
        .patch_pe01_out(patch_pe01_out),
        // ... etc for all patch outputs
        .patches_valid(patches_valid)
    );
*/ 
