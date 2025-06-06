// module unified_buffer_old #(
//     parameter MAX_IMG_W = 64,
//     parameter MAX_IMG_H = 64,
//     parameter BUFFER_SIZE = 7,
//     parameter PATCH_SIZE = 4,
//     parameter PATCHES_PER_BLOCK = 4,
//     parameter MAX_CHANNELS = 64,
//     parameter MAX_PADDING = 3  // Maximum padding size (e.g., for 7x7 kernels)
// )(
//     input logic clk,
//     input logic reset,
    
//     // Control signals
//     input logic start_extraction,
//     input logic next_channel_group,  // Request next 4 channels for same spatial block
//     input logic next_spatial_block,  // Request next spatial block after all channels done
    
//     // Configuration  
//     input logic [$clog2(MAX_IMG_W+1)-1:0] img_width,
//     input logic [$clog2(MAX_IMG_H+1)-1:0] img_height,
//     input logic [$clog2(MAX_CHANNELS+1)-1:0] num_channels,
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_top,     // Top padding
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_bottom,  // Bottom padding  
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_left,    // Left padding
//     input logic [$clog2(MAX_PADDING+1)-1:0] pad_right,   // Right padding
    
//     // Memory interface
//     output logic ram_re,
//     output logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr,
//     input logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3,
    
//     // Status outputs
//     output logic block_ready,
//     output logic extraction_complete,
//     output logic all_channels_done,  // All channel groups for current spatial block processed
//     output logic buffer_loading_complete, // Buffer is fully loaded with valid data
    
//     // Address calculation outputs (without padding offset)
//     output logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr,  // Block start column in image coordinates
//     output logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr,  // Block start row in image coordinates
//     output logic block_coords_valid,  // Indicates if the address coordinates are valid (not in padding)
    
//     // Data outputs: All 7x7 positions for spatial streaming (not just 4x4 patches)
//     output logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out, patch_pe04_out, patch_pe05_out, patch_pe06_out,
//     output logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out, patch_pe14_out, patch_pe15_out, patch_pe16_out,
//     output logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out, patch_pe24_out, patch_pe25_out, patch_pe26_out,
//     output logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out, patch_pe34_out, patch_pe35_out, patch_pe36_out,
//     output logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out, patch_pe44_out, patch_pe45_out, patch_pe46_out,
//     output logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out, patch_pe54_out, patch_pe55_out, patch_pe56_out,
//     output logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out, patch_pe64_out, patch_pe65_out, patch_pe66_out,
//     output logic patches_valid
// );

//     // Calculate spatial positions per RAM read (16 bytes / num_channels)
//     logic [$clog2(17)-1:0] spatial_positions_per_read;
//     always_comb begin
//         case (num_channels)
//             1:  spatial_positions_per_read = 16;
//             2:  spatial_positions_per_read = 8;
//             4:  spatial_positions_per_read = 4;
//             8:  spatial_positions_per_read = 2;
//             16: spatial_positions_per_read = 1;
//             default: spatial_positions_per_read = 1; // Conservative fallback
//         endcase
//     end

//     // State machine
//     typedef enum logic [2:0] {
//         IDLE,
//         LOADING_BLOCK,
//         EXTRACTING_CHANNELS,  // New state for extracting channel groups from cached data
//         BLOCK_READY,
//         WAIT_NEXT_SPATIAL
//     } state_t;
//     state_t state;

//     // Current spatial block position (top-left corner of 7x7 block in padded coordinate space)
//     // Note: These can be negative due to padding
//     logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] block_start_col;
//     logic signed [$clog2(MAX_IMG_H + 2*MAX_PADDING)-1:0] block_start_row;
    
//     // Current channel group (which set of 4 channels: 0, 1, 2, ...)
//     logic [$clog2(MAX_CHANNELS/4)-1:0] channel_group;
//     logic [$clog2(MAX_CHANNELS/4)-1:0] total_channel_groups;
    
//     // Buffer to store current 7x7x4 block
//     logic [31:0] buffer_7x7 [6:0][6:0]; // 7x7 spatial positions, 4 channels each
    
//     // RAM data cache to store full 128-bit reads
//     logic [127:0] ram_data_cache;
//     logic ram_data_valid;
    
//     // Loading counters - now track RAM reads, not individual positions
//     logic [$clog2(BUFFER_SIZE*BUFFER_SIZE/16 + 1)-1:0] ram_read_count;  // Number of RAM reads needed
//     logic [$clog2(BUFFER_SIZE*BUFFER_SIZE/16 + 1)-1:0] total_ram_reads; // Total RAM reads for 7x7 buffer
//     logic [$clog2(BUFFER_SIZE)-1:0] extract_row, extract_col;           // Position being extracted
    
//     // Calculate total RAM reads needed for 7x7 buffer
//     always_comb begin
//         // Total positions in 7x7 buffer = 49
//         // Round up division: (49 + spatial_positions_per_read - 1) / spatial_positions_per_read
//         total_ram_reads = (49 + spatial_positions_per_read - 1) / spatial_positions_per_read;
//     end
    
//     // Padded image dimensions - properly sized to handle padding additions
//     logic [$clog2(MAX_IMG_W + 2*MAX_PADDING + 1)-1:0] padded_width;
//     logic [$clog2(MAX_IMG_H + 2*MAX_PADDING + 1)-1:0] padded_height;
    
//     assign padded_width = ($clog2(MAX_IMG_W + 2*MAX_PADDING + 1))'(img_width) + ($clog2(MAX_IMG_W + 2*MAX_PADDING + 1))'(pad_left) + ($clog2(MAX_IMG_W + 2*MAX_PADDING + 1))'(pad_right);
//     assign padded_height = ($clog2(MAX_IMG_H + 2*MAX_PADDING + 1))'(img_height) + ($clog2(MAX_IMG_H + 2*MAX_PADDING + 1))'(pad_top) + ($clog2(MAX_IMG_H + 2*MAX_PADDING + 1))'(pad_bottom);
    
//     // Calculate total channel groups needed - fix width truncation
//     assign total_channel_groups = ($clog2(MAX_CHANNELS/4))'((32'(num_channels) + 3) / 4); // Ceiling division
    
//     // Calculate current extraction position in 7x7 buffer based on RAM read progress
//     logic [$clog2(BUFFER_SIZE*BUFFER_SIZE)-1:0] linear_position;
//     assign linear_position = ram_read_count * spatial_positions_per_read;
//     assign extract_row = linear_position / BUFFER_SIZE;
//     assign extract_col = linear_position % BUFFER_SIZE;
    
//     // State machine for spatial-channel iteration
//     always_ff @(posedge clk) begin
//         if (reset) begin
//             state <= IDLE;
//             block_start_row <= -($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);  // Start with negative row (top padding)
//             block_start_col <= -($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left); // Start with negative col (left padding)
//             channel_group <= 0;
//             ram_read_count <= 0;
//             ram_data_valid <= 0;
//         end else begin
//             case (state)
//                 IDLE: begin
//                     if (start_extraction) begin
//                         block_start_row <= -($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);  // Start with top padding
//                         block_start_col <= -($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left); // Start with left padding
//                         channel_group <= 0;
//                         ram_read_count <= 0;
//                         ram_data_valid <= 0;
//                         state <= LOADING_BLOCK;
//                     end
//                 end
                
//                 LOADING_BLOCK: begin
//                     if (ram_data_valid) begin
//                         // Check if we have loaded all required RAM reads for the 7x7 buffer
//                         if (ram_read_count >= total_ram_reads - 1) begin
//                             state <= BLOCK_READY;
//                         end else begin
//                             // Move to next RAM read
//                             ram_read_count <= ram_read_count + 1;
//                             ram_data_valid <= 1'b0;  // Reset to trigger next read
//                         end
//                     end
//                 end
                
//                 BLOCK_READY: begin
//                     if (next_channel_group) begin
//                         if (channel_group == total_channel_groups - 1) begin
//                             // All channels done for this spatial block
//                             channel_group <= 0;
//                             state <= WAIT_NEXT_SPATIAL;
//                         end else begin
//                             // Move to next channel group for same spatial block
//                             channel_group <= channel_group + 1;
//                             ram_read_count <= 0;
//                             ram_data_valid <= 0;
//                             state <= LOADING_BLOCK;
//                         end
//                     end
//                 end
                
//                 WAIT_NEXT_SPATIAL: begin
//                     if (next_spatial_block) begin
//                         // Move to next spatial block
//                         if (block_start_col + PATCHES_PER_BLOCK >= padded_width - PATCHES_PER_BLOCK) begin
//                             block_start_col <= -($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left); // Reset to left padding
//                             if (block_start_row + PATCHES_PER_BLOCK >= padded_height - PATCHES_PER_BLOCK) begin
//                                 // Extraction complete
//                                 state <= IDLE;
//                             end else begin
//                                 block_start_row <= block_start_row + PATCHES_PER_BLOCK;
//                                 ram_read_count <= 0;
//                                 ram_data_valid <= 0;
//                                 state <= LOADING_BLOCK;
//                             end
//                         end else begin
//                             block_start_col <= block_start_col + PATCHES_PER_BLOCK;
//                             ram_read_count <= 0;
//                             ram_data_valid <= 0;
//                             state <= LOADING_BLOCK;
//                         end
//                     end
//                 end
                
//                 default: begin
//                     state <= IDLE;
//                 end
//             endcase
//         end
//     end
    
//     // Calculate memory address with overflow protection
//     logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] calculated_addr;
//     logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] max_valid_addr;
    
//     // Temporary variable for address calculation (declared outside always_comb to prevent latches)
//     logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8-1:0] temp_addr;
    
//     // Calculate base memory address for current RAM read
//     // The address corresponds to the first spatial position in this RAM read
//     logic [$clog2(MAX_IMG_W)-1:0] base_mem_row, base_mem_col;
//     logic base_address_valid;
//     logic [$clog2(BUFFER_SIZE*BUFFER_SIZE)-1:0] base_linear_pos;
    
//     assign base_linear_pos = ram_read_count * spatial_positions_per_read;
//     logic [$clog2(BUFFER_SIZE)-1:0] base_buf_row, base_buf_col;
//     assign base_buf_row = base_linear_pos / BUFFER_SIZE;
//     assign base_buf_col = base_linear_pos % BUFFER_SIZE;
    
//     // Calculate actual coordinates for base position
//     logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] base_actual_row, base_actual_col;
//     assign base_actual_row = block_start_row + ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(base_buf_row);
//     assign base_actual_col = block_start_col + ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(base_buf_col);
    
//     // Check if base position is in padding region
//     logic base_in_padding_region;
//     assign base_in_padding_region = (base_actual_row < ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top)) ||
//                                    (base_actual_row >= (($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top) + ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(img_height))) ||
//                                    (base_actual_col < ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left)) ||
//                                    (base_actual_col >= (($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left) + ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(img_width)));
    
//     always_comb begin
//         // Default values to prevent latches
//         calculated_addr = '0;
//         temp_addr = '0;
//         base_mem_row = '0;
//         base_mem_col = '0;
//         base_address_valid = 1'b0;
        
//         // Calculate maximum valid address - fix to account for correct addressing
//         // Each address contains data for all channel groups at one spatial position
//         max_valid_addr = ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4))'((img_width * img_height) - 1);
        
//         if (!base_in_padding_region) begin
//             // Convert base position to memory coordinates
//             logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] temp_row, temp_col;
//             temp_row = base_actual_row - ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);
//             temp_col = base_actual_col - ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left);
            
//             // Ensure coordinates are within valid range
//             if (temp_row >= 0 && temp_row < img_height && temp_col >= 0 && temp_col < img_width) begin
//                 base_mem_row = ($clog2(MAX_IMG_W))'(temp_row);
//                 base_mem_col = ($clog2(MAX_IMG_W))'(temp_col);
//                 base_address_valid = 1'b1;
                
//                 // Calculate address based on spatial_positions_per_read
//                 // Address needs to be adjusted to account for the memory packing
//                 temp_addr = (($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(base_mem_row) * ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(img_width) + ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(base_mem_col)) / ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(spatial_positions_per_read);
                
//                 if (temp_addr <= ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(max_valid_addr)) begin
//                     calculated_addr = temp_addr[$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0];
//                 end
//             end
//         end
//     end
    
//     assign ram_addr = calculated_addr;
//     assign ram_re = (state == LOADING_BLOCK) && !ram_data_valid && base_address_valid && !base_in_padding_region;
    
//     // Address bounds checking
//     always_ff @(posedge clk) begin
//         if (ram_re && calculated_addr > max_valid_addr) begin
//             $display("ERROR: unified_buffer ram address out of bounds at time %0t! calculated_addr=%d, max_valid_addr=%d, base_mem_row=%d, base_mem_col=%d, img_width=%d, img_height=%d", 
//                      $time, calculated_addr, max_valid_addr, base_mem_row, base_mem_col, img_width, img_height);
//         end
//     end
    
//     // RAM data caching and validation
//     always_ff @(posedge clk) begin
//         if (reset) begin
//             ram_data_cache <= '0;
//             ram_data_valid <= 1'b0;
//         end else begin
//             // Cache RAM data when read enable is active (with 1 cycle delay for RAM output)
//             if (ram_re) begin
//                 // Next cycle, cache the RAM output
//                 ram_data_cache <= {ram_dout0, ram_dout1, ram_dout2, ram_dout3};
//                 ram_data_valid <= 1'b1;
//             end else if (state == IDLE || state == WAIT_NEXT_SPATIAL || 
//                         (state == BLOCK_READY && next_channel_group && channel_group < total_channel_groups - 1)) begin
//                 // Reset cache when starting new operations
//                 ram_data_valid <= 1'b0;
//             end
//         end
//     end
    
//     // Extract multiple spatial positions from cached RAM data
//     // This function extracts the correct channel data for a given spatial position index
//     function automatic [31:0] extract_spatial_position(
//         input [127:0] cached_data,
//         input [$clog2(17)-1:0] pos_index,
//         input [1:0] ch_group,
//         input [$clog2(17)-1:0] positions_per_read,
//         input [$clog2(MAX_CHANNELS+1)-1:0] num_ch
//     );
//         logic [31:0] result;
//         result = 32'h00000000; // Default to zero
        
//         case (positions_per_read)
//             16: begin // 1 channel per position, 16 positions per read
//                 if (ch_group == 2'd0) begin
//                     // Each position has only 1 channel, put it in lower 8 bits
//                     result = {24'b0, cached_data[(pos_index * 8) +: 8]};
//                 end
//             end
//             8: begin // 2 channels per position, 8 positions per read  
//                 if (ch_group == 2'd0) begin
//                     // Each position has 2 channels, put them in lower 16 bits
//                     result = {16'b0, cached_data[(pos_index * 16) +: 16]};
//                 end
//             end
//             4: begin // 4 channels per position, 4 positions per read
//                 if (ch_group == 2'd0) begin
//                     // Each position has exactly 4 channels, use all 32 bits
//                     result = cached_data[(pos_index * 32) +: 32];
//                 end
//             end
//             2: begin // 8 channels per position, 2 positions per read
//                 // Each position has 8 channels, extract 4 channels based on channel group
//                 case (ch_group)
//                     2'd0: result = cached_data[(pos_index * 64) +: 32];        // Channels 0-3 of this position
//                     2'd1: result = cached_data[(pos_index * 64 + 32) +: 32];   // Channels 4-7 of this position
//                 endcase
//             end
//             1: begin // 16 channels per position, 1 position per read
//                 // Single position has 16 channels, extract 4 channels based on channel group
//                 case (ch_group)
//                     2'd0: result = cached_data[31:0];   // Channels 0-3
//                     2'd1: result = cached_data[63:32];  // Channels 4-7
//                     2'd2: result = cached_data[95:64];  // Channels 8-11
//                     2'd3: result = cached_data[127:96]; // Channels 12-15
//                 endcase
//             end
//         endcase
//         return result;
//     endfunction
    
//     // Store extracted data into 7x7 buffer
//     logic [$clog2(17)-1:0] spatial_pos_in_read;
//     always_ff @(posedge clk) begin
//         if (ram_data_valid && state == LOADING_BLOCK) begin
//             // Extract all spatial positions from current RAM read
//             for (int pos = 0; pos < spatial_positions_per_read; pos++) begin
//                 logic [$clog2(BUFFER_SIZE)-1:0] buf_row, buf_col;
//                 logic [$clog2(BUFFER_SIZE*BUFFER_SIZE)-1:0] abs_pos;
//                 logic has_valid_channels;
                
//                 abs_pos = ram_read_count * spatial_positions_per_read + pos;
//                 buf_row = abs_pos / BUFFER_SIZE;
//                 buf_col = abs_pos % BUFFER_SIZE;
                
//                 // Check if this spatial position has valid channels for the current channel group
//                 case (spatial_positions_per_read)
//                     16, 8, 4: has_valid_channels = (channel_group == 0); // Only channel group 0 is valid
//                     2: has_valid_channels = (channel_group <= 1); // Channel groups 0 and 1 are valid
//                     1: has_valid_channels = (channel_group <= 3); // All channel groups 0-3 are valid
//                     default: has_valid_channels = 1'b0;
//                 endcase
                
//                 // Only store if within 7x7 buffer bounds and has valid channels for this channel group
//                 if (buf_row < BUFFER_SIZE && buf_col < BUFFER_SIZE && has_valid_channels) begin
//                     // Check if this position is in padding
//                     logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] pos_actual_row, pos_actual_col;
//                     logic pos_in_padding;
                    
//                     pos_actual_row = block_start_row + ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(buf_row);
//                     pos_actual_col = block_start_col + ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(buf_col);
                    
//                     pos_in_padding = (pos_actual_row < ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top)) ||
//                                    (pos_actual_row >= (($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top) + ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(img_height))) ||
//                                    (pos_actual_col < ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left)) ||
//                                    (pos_actual_col >= (($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left) + ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(img_width)));
                    
//                     if (pos_in_padding) begin
//                         buffer_7x7[buf_row][buf_col] <= 32'h00000000;
//                     end else begin
//                         buffer_7x7[buf_row][buf_col] <= extract_spatial_position(ram_data_cache, pos[$clog2(17)-1:0], channel_group[1:0], spatial_positions_per_read, num_channels);
//                     end
//                 end
//             end
//         end
//     end
    
//     // Extract 16 overlapping 4x4 patches from the 7x7 buffer
//     // Patch (i,j) starts at buffer position (i,j) and extends 4x4
//     assign patch_pe00_out = buffer_7x7[0][0]; // Patch at (0,0)
//     assign patch_pe01_out = buffer_7x7[0][1]; // Patch at (0,1)  
//     assign patch_pe02_out = buffer_7x7[0][2]; // Patch at (0,2)
//     assign patch_pe03_out = buffer_7x7[0][3]; // Patch at (0,3)
    
//     assign patch_pe10_out = buffer_7x7[1][0]; // Patch at (1,0)
//     assign patch_pe11_out = buffer_7x7[1][1]; // Patch at (1,1)
//     assign patch_pe12_out = buffer_7x7[1][2]; // Patch at (1,2) 
//     assign patch_pe13_out = buffer_7x7[1][3]; // Patch at (1,3)
    
//     assign patch_pe20_out = buffer_7x7[2][0]; // Patch at (2,0)
//     assign patch_pe21_out = buffer_7x7[2][1]; // Patch at (2,1)
//     assign patch_pe22_out = buffer_7x7[2][2]; // Patch at (2,2)
//     assign patch_pe23_out = buffer_7x7[2][3]; // Patch at (2,3)
    
//     assign patch_pe30_out = buffer_7x7[3][0]; // Patch at (3,0)
//     assign patch_pe31_out = buffer_7x7[3][1]; // Patch at (3,1)
//     assign patch_pe32_out = buffer_7x7[3][2]; // Patch at (3,2)
//     assign patch_pe33_out = buffer_7x7[3][3]; // Patch at (3,3)
    
//     assign patch_pe40_out = buffer_7x7[4][0]; // Patch at (4,0)
//     assign patch_pe41_out = buffer_7x7[4][1]; // Patch at (4,1)
//     assign patch_pe42_out = buffer_7x7[4][2]; // Patch at (4,2)
//     assign patch_pe43_out = buffer_7x7[4][3]; // Patch at (4,3)
    
//     assign patch_pe50_out = buffer_7x7[5][0]; // Patch at (5,0)
//     assign patch_pe51_out = buffer_7x7[5][1]; // Patch at (5,1)
//     assign patch_pe52_out = buffer_7x7[5][2]; // Patch at (5,2)
//     assign patch_pe53_out = buffer_7x7[5][3]; // Patch at (5,3)
    
//     assign patch_pe60_out = buffer_7x7[6][0]; // Patch at (6,0)
//     assign patch_pe61_out = buffer_7x7[6][1]; // Patch at (6,1)
//     assign patch_pe62_out = buffer_7x7[6][2]; // Patch at (6,2)
//     assign patch_pe63_out = buffer_7x7[6][3]; // Patch at (6,3)
    
//     assign patch_pe04_out = buffer_7x7[0][4]; // Patch at (0,4)
//     assign patch_pe05_out = buffer_7x7[0][5]; // Patch at (0,5)
//     assign patch_pe06_out = buffer_7x7[0][6]; // Patch at (0,6)
    
//     assign patch_pe14_out = buffer_7x7[1][4]; // Patch at (1,4)
//     assign patch_pe15_out = buffer_7x7[1][5]; // Patch at (1,5)
//     assign patch_pe16_out = buffer_7x7[1][6]; // Patch at (1,6)
    
//     assign patch_pe24_out = buffer_7x7[2][4]; // Patch at (2,4)
//     assign patch_pe25_out = buffer_7x7[2][5]; // Patch at (2,5)
//     assign patch_pe26_out = buffer_7x7[2][6]; // Patch at (2,6)
    
//     assign patch_pe34_out = buffer_7x7[3][4]; // Patch at (3,4)
//     assign patch_pe35_out = buffer_7x7[3][5]; // Patch at (3,5)
//     assign patch_pe36_out = buffer_7x7[3][6]; // Patch at (3,6)
    
//     assign patch_pe44_out = buffer_7x7[4][4]; // Patch at (4,4)
//     assign patch_pe45_out = buffer_7x7[4][5]; // Patch at (4,5)
//     assign patch_pe46_out = buffer_7x7[4][6]; // Patch at (4,6)
    
//     assign patch_pe54_out = buffer_7x7[5][4]; // Patch at (5,4)
//     assign patch_pe55_out = buffer_7x7[5][5]; // Patch at (5,5)
//     assign patch_pe56_out = buffer_7x7[5][6]; // Patch at (5,6)
    
//     assign patch_pe64_out = buffer_7x7[6][4]; // Patch at (6,4)
//     assign patch_pe65_out = buffer_7x7[6][5]; // Patch at (6,5)
//     assign patch_pe66_out = buffer_7x7[6][6]; // Patch at (6,6)
    
//     // Status signals
//     assign block_ready = (state == BLOCK_READY) && ram_data_valid;
//     assign patches_valid = (state == BLOCK_READY) && ram_data_valid;
//     assign extraction_complete = (state == IDLE) && (32'(block_start_row) >= 32'(padded_height) - PATCHES_PER_BLOCK);
//     assign all_channels_done = (state == WAIT_NEXT_SPATIAL);
//     assign buffer_loading_complete = ram_data_valid;
    
//     // Calculate address coordinates (without padding offset)
//     logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] image_start_col, image_start_row;
    
//     // Convert padded coordinates to image coordinates
//     assign image_start_col = block_start_col + ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left);
//     assign image_start_row = block_start_row + ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);
    
//     // Check if block start is within valid image bounds and convert to unsigned
//     assign block_coords_valid = (image_start_col >= 0) && 
//                                (image_start_col < img_width) && 
//                                (image_start_row >= 0) && 
//                                (image_start_row < img_height);
    
//     assign block_start_col_addr = block_coords_valid ? ($clog2(MAX_IMG_W))'(image_start_col) : '0;
//     assign block_start_row_addr = block_coords_valid ? ($clog2(MAX_IMG_H))'(image_start_row) : '0;

// endmodule 
