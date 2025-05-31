module unified_buffer #(
    parameter MAX_IMG_W = 64,
    parameter MAX_IMG_H = 64,
    parameter BUFFER_SIZE = 7,
    parameter PATCH_SIZE = 4,
    parameter PATCHES_PER_BLOCK = 4,
    parameter MAX_CHANNELS = 64,
    parameter MAX_PADDING = 3  // Maximum padding size (e.g., for 7x7 kernels)
)(
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_extraction,
    input logic next_channel_group,  // Request next 4 channels for same spatial block
    input logic next_spatial_block,  // Request next spatial block after all channels done
    
    // Configuration  
    input logic [$clog2(MAX_IMG_W+1)-1:0] img_width,
    input logic [$clog2(MAX_IMG_H+1)-1:0] img_height,
    input logic [$clog2(MAX_CHANNELS+1)-1:0] num_channels,
    input logic [$clog2(MAX_PADDING+1)-1:0] pad_top,     // Top padding
    input logic [$clog2(MAX_PADDING+1)-1:0] pad_bottom,  // Bottom padding  
    input logic [$clog2(MAX_PADDING+1)-1:0] pad_left,    // Left padding
    input logic [$clog2(MAX_PADDING+1)-1:0] pad_right,   // Right padding
    
    // Memory interface
    output logic ram_re,
    output logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr,
    input logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3,
    
    // Status outputs
    output logic block_ready,
    output logic extraction_complete,
    output logic all_channels_done,  // All channel groups for current spatial block processed
    output logic buffer_loading_complete, // Buffer is fully loaded with valid data
    
    // Data outputs: All 7x7 positions for spatial streaming (not just 4x4 patches)
    output logic [31:0] patch_pe00_out, patch_pe01_out, patch_pe02_out, patch_pe03_out, patch_pe04_out, patch_pe05_out, patch_pe06_out,
    output logic [31:0] patch_pe10_out, patch_pe11_out, patch_pe12_out, patch_pe13_out, patch_pe14_out, patch_pe15_out, patch_pe16_out,
    output logic [31:0] patch_pe20_out, patch_pe21_out, patch_pe22_out, patch_pe23_out, patch_pe24_out, patch_pe25_out, patch_pe26_out,
    output logic [31:0] patch_pe30_out, patch_pe31_out, patch_pe32_out, patch_pe33_out, patch_pe34_out, patch_pe35_out, patch_pe36_out,
    output logic [31:0] patch_pe40_out, patch_pe41_out, patch_pe42_out, patch_pe43_out, patch_pe44_out, patch_pe45_out, patch_pe46_out,
    output logic [31:0] patch_pe50_out, patch_pe51_out, patch_pe52_out, patch_pe53_out, patch_pe54_out, patch_pe55_out, patch_pe56_out,
    output logic [31:0] patch_pe60_out, patch_pe61_out, patch_pe62_out, patch_pe63_out, patch_pe64_out, patch_pe65_out, patch_pe66_out,
    output logic patches_valid
);

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        LOADING_BLOCK,
        BLOCK_READY,
        WAIT_NEXT_CHANNEL,
        WAIT_NEXT_SPATIAL
    } state_t;
    state_t state;

    // Current spatial block position (top-left corner of 7x7 block in padded coordinate space)
    // Note: These can be negative due to padding
    logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] block_start_col;
    logic signed [$clog2(MAX_IMG_H + 2*MAX_PADDING)-1:0] block_start_row;
    
    // Current channel group (which set of 4 channels: 0, 1, 2, ...)
    logic [$clog2(MAX_CHANNELS/4)-1:0] channel_group;
    logic [$clog2(MAX_CHANNELS/4)-1:0] total_channel_groups;
    
    // Buffer to store current 7x7x4 block
    logic [31:0] buffer_7x7 [6:0][6:0]; // 7x7 spatial positions, 4 channels each
    
    // Loading counters
    logic [$clog2(BUFFER_SIZE)-1:0] load_row, load_col;
    logic loading_cycle_complete;
    logic all_data_loaded;
    
    // Padded image dimensions - properly sized to handle padding additions
    logic [$clog2(MAX_IMG_W + 2*MAX_PADDING + 1)-1:0] padded_width;
    logic [$clog2(MAX_IMG_H + 2*MAX_PADDING + 1)-1:0] padded_height;
    
    assign padded_width = ($clog2(MAX_IMG_W + 2*MAX_PADDING + 1))'(img_width) + ($clog2(MAX_IMG_W + 2*MAX_PADDING + 1))'(pad_left) + ($clog2(MAX_IMG_W + 2*MAX_PADDING + 1))'(pad_right);
    assign padded_height = ($clog2(MAX_IMG_H + 2*MAX_PADDING + 1))'(img_height) + ($clog2(MAX_IMG_H + 2*MAX_PADDING + 1))'(pad_top) + ($clog2(MAX_IMG_H + 2*MAX_PADDING + 1))'(pad_bottom);
    
    // Calculate total channel groups needed - fix width truncation
    assign total_channel_groups = ($clog2(MAX_CHANNELS/4))'((32'(num_channels) + 3) / 4); // Ceiling division
    
    // State machine for spatial-channel iteration
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            block_start_row <= -($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);  // Start with negative row (top padding)
            block_start_col <= -($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left); // Start with negative col (left padding)
            channel_group <= 0;
            load_row <= 0;
            load_col <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start_extraction) begin
                        block_start_row <= -($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);  // Start with top padding
                        block_start_col <= -($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left); // Start with left padding
                        channel_group <= 0;
                        load_row <= 0;
                        load_col <= 0;
                        state <= LOADING_BLOCK;
                    end
                end
                
                LOADING_BLOCK: begin
                    if (loading_complete) begin
                        state <= BLOCK_READY;
                    end else begin
                        // Increment loading counters
                        if (load_col == BUFFER_SIZE - 1) begin
                            load_col <= 0;
                            if (load_row == BUFFER_SIZE - 1) begin
                                load_row <= 0;
                            end else begin
                                load_row <= load_row + 1;
                            end
                        end else begin
                            load_col <= load_col + 1;
                        end
                    end
                end
                
                BLOCK_READY: begin
                    if (next_channel_group) begin
                        if (channel_group == total_channel_groups - 1) begin
                            // All channels done for this spatial block
                            channel_group <= 0;
                            state <= WAIT_NEXT_SPATIAL;
                        end else begin
                            // Move to next channel group for same spatial block
                            channel_group <= channel_group + 1;
                            load_row <= 0;
                            load_col <= 0;
                            state <= LOADING_BLOCK;
                        end
                    end
                end
                
                WAIT_NEXT_SPATIAL: begin
                    if (next_spatial_block) begin
                        // Move to next spatial block
                        if (block_start_col + PATCHES_PER_BLOCK >= padded_width - PATCHES_PER_BLOCK) begin
                            block_start_col <= -($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left); // Reset to left padding
                            if (block_start_row + PATCHES_PER_BLOCK >= padded_height - PATCHES_PER_BLOCK) begin
                                // Extraction complete
                                state <= IDLE;
                            end else begin
                                block_start_row <= block_start_row + PATCHES_PER_BLOCK;
                                load_row <= 0;
                                load_col <= 0;
                                state <= LOADING_BLOCK;
                            end
                        end else begin
                            block_start_col <= block_start_col + PATCHES_PER_BLOCK;
                            load_row <= 0;
                            load_col <= 0;
                            state <= LOADING_BLOCK;
                        end
                    end
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
    
    // Loading completion detection - simplified and more reliable
    logic loading_complete;
    logic [$clog2(8)-1:0] completion_delay_counter;
    
    assign loading_cycle_complete = (load_row == BUFFER_SIZE - 1) && (load_col == BUFFER_SIZE - 1);
    
    // Track that all required data has been loaded (including delayed RAM data)
    always_ff @(posedge clk) begin
        if (reset) begin
            all_data_loaded <= 1'b0;
            completion_delay_counter <= 0;
        end else begin
            if (state == LOADING_BLOCK && loading_cycle_complete) begin
                // Start delay counter to wait for RAM data
                completion_delay_counter <= completion_delay_counter + 1;
                if (completion_delay_counter >= 2) begin // Wait 2 cycles for RAM data to settle
                    all_data_loaded <= 1'b1;
                end
            end else if (state == IDLE || state == WAIT_NEXT_SPATIAL || 
                        (state == BLOCK_READY && next_channel_group && channel_group < total_channel_groups - 1)) begin
                // Reset when:
                // 1. Starting a new block (IDLE)
                // 2. Waiting for next spatial block
                // 3. Starting next channel group (transition from BLOCK_READY)
                all_data_loaded <= 1'b0;
                completion_delay_counter <= 0;
            end
        end
    end
    
    assign loading_complete = all_data_loaded;
    
    // Memory address calculation with improved padding support and bounds checking
    logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] actual_row, actual_col;
    logic in_padding_region;
    logic [$clog2(MAX_IMG_W)-1:0] mem_row, mem_col;
    logic address_valid;
    
    // Temporary variables for coordinate conversion (declared outside always_comb to prevent latches)
    logic signed [$clog2(MAX_IMG_W + 2*MAX_PADDING)-1:0] temp_row, temp_col;
    
    // Calculate actual coordinates in padded space
    assign actual_row = block_start_row + ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(load_row);
    assign actual_col = block_start_col + ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(load_col);
    
    // Check if we're in padding region (outside actual image bounds)
    assign in_padding_region = (actual_row < ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top)) ||
                              (actual_row >= ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top + img_height)) ||
                              (actual_col < ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left)) ||
                              (actual_col >= ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left + img_width));
    
    // Convert to memory coordinates (relative to actual image, not padded) with bounds checking
    always_comb begin
        // Default values to prevent latches
        mem_row = '0;
        mem_col = '0;
        address_valid = 1'b0;
        temp_row = '0;
        temp_col = '0;
        
        if (!in_padding_region) begin
            // Safe conversion with bounds checking
            temp_row = actual_row - ($clog2(MAX_IMG_H + 2*MAX_PADDING))'(pad_top);
            temp_col = actual_col - ($clog2(MAX_IMG_W + 2*MAX_PADDING))'(pad_left);
            
            // Ensure coordinates are within valid range
            if (temp_row >= 0 && temp_row < img_height && temp_col >= 0 && temp_col < img_width) begin
                mem_row = ($clog2(MAX_IMG_W))'(temp_row);
                mem_col = ($clog2(MAX_IMG_W))'(temp_col);
                address_valid = 1'b1;
            end
        end
    end
    
    // Calculate memory address with overflow protection
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] calculated_addr;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] max_valid_addr;
    
    // Temporary variable for address calculation (declared outside always_comb to prevent latches)
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8-1:0] temp_addr;
    
    always_comb begin
        // Default values to prevent latches
        calculated_addr = '0;
        temp_addr = '0;
        
        // Calculate maximum valid address - fix to account for correct addressing
        // Each address contains data for all channel groups at one spatial position
        max_valid_addr = ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4))'((img_width * img_height) - 1);
        
        if (address_valid && !in_padding_region) begin
            // Fixed address calculation: all channel groups for same position use same address
            // Address = row * width + col (no multiplication by channel groups)
            // The channel group determines which 32-bit chunk to extract from 128-bit data
            temp_addr = ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(mem_row) * ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(img_width) + ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(mem_col);
            
            if (temp_addr <= ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)+8)'(max_valid_addr)) begin
                calculated_addr = temp_addr[$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0];
            end
        end
    end
    
    assign ram_addr = calculated_addr;
    assign ram_re = (state == LOADING_BLOCK) && !loading_cycle_complete && address_valid && !in_padding_region;
    
    // Store loaded data into 7x7 buffer
    // Need to delay by one cycle to properly capture RAM data
    logic ram_re_delayed;
    logic in_padding_region_delayed;
    logic [$clog2(BUFFER_SIZE)-1:0] load_row_delayed, load_col_delayed;
    logic address_valid_delayed;
    logic [1:0] channel_group_delayed;
    
    always_ff @(posedge clk) begin
        // Delay control signals by one cycle to match RAM data timing
        ram_re_delayed <= ram_re;
        in_padding_region_delayed <= in_padding_region;
        load_row_delayed <= load_row;
        load_col_delayed <= load_col;
        address_valid_delayed <= address_valid;
        channel_group_delayed <= channel_group;
    end
    
    always_ff @(posedge clk) begin
        if (ram_re_delayed) begin
            if (in_padding_region_delayed) begin
                // Zero padding for regions outside image bounds
                buffer_7x7[load_row_delayed][load_col_delayed] <= 32'h00000000;
            end else begin
                // Select the correct 32-bit chunk based on channel group and reverse byte order
                // The tensor RAM outputs are in little-endian format, so we need to reverse bytes
                // Channel group 0 (channels 0-3): use ram_dout3 with byte reversal
                // Channel group 1 (channels 4-7): use ram_dout2 with byte reversal
                case (channel_group_delayed)
                    2'd0: buffer_7x7[load_row_delayed][load_col_delayed] <= {ram_dout3[7:0], ram_dout3[15:8], ram_dout3[23:16], ram_dout3[31:24]}; // bytes [0,1,2,3]
                    2'd1: buffer_7x7[load_row_delayed][load_col_delayed] <= {ram_dout2[7:0], ram_dout2[15:8], ram_dout2[23:16], ram_dout2[31:24]}; // bytes [4,5,6,7]
                    2'd2: buffer_7x7[load_row_delayed][load_col_delayed] <= {ram_dout1[7:0], ram_dout1[15:8], ram_dout1[23:16], ram_dout1[31:24]}; // bytes [8,9,10,11]
                    2'd3: buffer_7x7[load_row_delayed][load_col_delayed] <= {ram_dout0[7:0], ram_dout0[15:8], ram_dout0[23:16], ram_dout0[31:24]}; // bytes [12,13,14,15]
                    default: buffer_7x7[load_row_delayed][load_col_delayed] <= {ram_dout3[7:0], ram_dout3[15:8], ram_dout3[23:16], ram_dout3[31:24]};
                endcase
            end
        end else if (state == LOADING_BLOCK && in_padding_region) begin
            // Handle immediate padding (for regions that never generate RAM requests)
            buffer_7x7[load_row][load_col] <= 32'h00000000;
        end
    end
    
    // Extract 16 overlapping 4x4 patches from the 7x7 buffer
    // Patch (i,j) starts at buffer position (i,j) and extends 4x4
    assign patch_pe00_out = buffer_7x7[0][0]; // Patch at (0,0)
    assign patch_pe01_out = buffer_7x7[0][1]; // Patch at (0,1)  
    assign patch_pe02_out = buffer_7x7[0][2]; // Patch at (0,2)
    assign patch_pe03_out = buffer_7x7[0][3]; // Patch at (0,3)
    
    assign patch_pe10_out = buffer_7x7[1][0]; // Patch at (1,0)
    assign patch_pe11_out = buffer_7x7[1][1]; // Patch at (1,1)
    assign patch_pe12_out = buffer_7x7[1][2]; // Patch at (1,2) 
    assign patch_pe13_out = buffer_7x7[1][3]; // Patch at (1,3)
    
    assign patch_pe20_out = buffer_7x7[2][0]; // Patch at (2,0)
    assign patch_pe21_out = buffer_7x7[2][1]; // Patch at (2,1)
    assign patch_pe22_out = buffer_7x7[2][2]; // Patch at (2,2)
    assign patch_pe23_out = buffer_7x7[2][3]; // Patch at (2,3)
    
    assign patch_pe30_out = buffer_7x7[3][0]; // Patch at (3,0)
    assign patch_pe31_out = buffer_7x7[3][1]; // Patch at (3,1)
    assign patch_pe32_out = buffer_7x7[3][2]; // Patch at (3,2)
    assign patch_pe33_out = buffer_7x7[3][3]; // Patch at (3,3)
    
    assign patch_pe40_out = buffer_7x7[4][0]; // Patch at (4,0)
    assign patch_pe41_out = buffer_7x7[4][1]; // Patch at (4,1)
    assign patch_pe42_out = buffer_7x7[4][2]; // Patch at (4,2)
    assign patch_pe43_out = buffer_7x7[4][3]; // Patch at (4,3)
    
    assign patch_pe50_out = buffer_7x7[5][0]; // Patch at (5,0)
    assign patch_pe51_out = buffer_7x7[5][1]; // Patch at (5,1)
    assign patch_pe52_out = buffer_7x7[5][2]; // Patch at (5,2)
    assign patch_pe53_out = buffer_7x7[5][3]; // Patch at (5,3)
    
    assign patch_pe60_out = buffer_7x7[6][0]; // Patch at (6,0)
    assign patch_pe61_out = buffer_7x7[6][1]; // Patch at (6,1)
    assign patch_pe62_out = buffer_7x7[6][2]; // Patch at (6,2)
    assign patch_pe63_out = buffer_7x7[6][3]; // Patch at (6,3)
    
    assign patch_pe04_out = buffer_7x7[0][4]; // Patch at (0,4)
    assign patch_pe05_out = buffer_7x7[0][5]; // Patch at (0,5)
    assign patch_pe06_out = buffer_7x7[0][6]; // Patch at (0,6)
    
    assign patch_pe14_out = buffer_7x7[1][4]; // Patch at (1,4)
    assign patch_pe15_out = buffer_7x7[1][5]; // Patch at (1,5)
    assign patch_pe16_out = buffer_7x7[1][6]; // Patch at (1,6)
    
    assign patch_pe24_out = buffer_7x7[2][4]; // Patch at (2,4)
    assign patch_pe25_out = buffer_7x7[2][5]; // Patch at (2,5)
    assign patch_pe26_out = buffer_7x7[2][6]; // Patch at (2,6)
    
    assign patch_pe34_out = buffer_7x7[3][4]; // Patch at (3,4)
    assign patch_pe35_out = buffer_7x7[3][5]; // Patch at (3,5)
    assign patch_pe36_out = buffer_7x7[3][6]; // Patch at (3,6)
    
    assign patch_pe44_out = buffer_7x7[4][4]; // Patch at (4,4)
    assign patch_pe45_out = buffer_7x7[4][5]; // Patch at (4,5)
    assign patch_pe46_out = buffer_7x7[4][6]; // Patch at (4,6)
    
    assign patch_pe54_out = buffer_7x7[5][4]; // Patch at (5,4)
    assign patch_pe55_out = buffer_7x7[5][5]; // Patch at (5,5)
    assign patch_pe56_out = buffer_7x7[5][6]; // Patch at (5,6)
    
    assign patch_pe64_out = buffer_7x7[6][4]; // Patch at (6,4)
    assign patch_pe65_out = buffer_7x7[6][5]; // Patch at (6,5)
    assign patch_pe66_out = buffer_7x7[6][6]; // Patch at (6,6)
    
    // Status signals
    assign block_ready = (state == BLOCK_READY) && loading_complete;
    assign patches_valid = (state == BLOCK_READY) && loading_complete;
    assign extraction_complete = (state == IDLE) && (32'(block_start_row) >= 32'(padded_height) - PATCHES_PER_BLOCK);
    assign all_channels_done = (state == WAIT_NEXT_SPATIAL);
    assign buffer_loading_complete = loading_complete;

endmodule 
