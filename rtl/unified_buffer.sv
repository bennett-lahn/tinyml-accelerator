module unified_buffer #(
    parameter MAX_IMG_W = 64,
    parameter MAX_IMG_H = 64,
    parameter MAX_CHANNELS = 64,
    parameter MAX_PADDING = 3
)(
    input logic clk,
    input logic reset,
    
    // Layer configuration - determines extraction strategy
    input logic [2:0] layer_idx,  // 0=input, 1=conv1, 2=conv2, 3=conv3, 4=conv4
    
    // Control signals
    input logic start_extraction,
    input logic next_channel_group,
    input logic next_spatial_block,
    
    // Memory interface
    output logic ram_re,
    output logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr,
    input logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3,
    
    // Status outputs
    output logic block_ready,
    output logic extraction_complete,
    output logic all_channels_done,
    output logic buffer_loading_complete,
    
    // Address calculation outputs
    output logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr,
    output logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr,
    output logic block_coords_valid,
    
    // Data outputs - variable patch sizes based on layer
    output logic [31:0] patch_data_out [6:0][6:0],  // Maximum 7x7 buffer
    output logic patches_valid
);

    // Layer-specific configuration
    logic [$clog2(MAX_IMG_W+1)-1:0] layer_img_width;
    logic [$clog2(MAX_IMG_H+1)-1:0] layer_img_height;
    logic [$clog2(MAX_CHANNELS+1)-1:0] layer_num_channels;
    logic [2:0] layer_patch_size;     // Size of patch to extract (4x4, 7x7, etc.)
    logic [2:0] layer_stride;         // Stride for spatial advancement
    logic layer_use_full_image;       // For small images, extract entire image
    // Add padding configuration using model's default asymmetric padding
    logic [2:0] layer_pad_top, layer_pad_bottom, layer_pad_left, layer_pad_right;
    logic [$clog2(MAX_IMG_W + 3 + 1)-1:0] padded_img_width;
    logic [$clog2(MAX_IMG_H + 3 + 1)-1:0] padded_img_height;

    // Hard-coded layer configurations based on TPU_target_model.py
    always_comb begin
        case (layer_idx)
            3'd0: begin // Input layer: 32x32x1
                layer_img_width = 32;
                layer_img_height = 32;
                layer_num_channels = 1;
                layer_patch_size = 7;
                layer_stride = 4;
                layer_use_full_image = 0;
                layer_pad_top = 0; layer_pad_bottom = 0; layer_pad_left = 0; layer_pad_right = 0;
            end
            3'd1: begin // After conv1+pool1: 16x16x8
                layer_img_width = 16;
                layer_img_height = 16;
                layer_num_channels = 8;
                layer_patch_size = 7;
                layer_stride = 4;
                layer_use_full_image = 0;
                layer_pad_top = 0; layer_pad_bottom = 0; layer_pad_left = 0; layer_pad_right = 0;
            end
            3'd2: begin // After conv2+pool2: 8x8x16 -> Pad to enable 7x7 chunks
                layer_img_width = 8;
                layer_img_height = 8;
                layer_num_channels = 16;
                layer_patch_size = 7;  // Use 7x7 chunks with padding
                layer_stride = 4;      // Keep stride=4, gives chunks at padded coordinates
                layer_use_full_image = 0;
                // Use model's default asymmetric padding: 8x8 -> 11x11
                layer_pad_top = 1; layer_pad_bottom = 2; 
                layer_pad_left = 1; layer_pad_right = 2;
            end
            3'd3: begin // After conv3+pool3: 4x4x32 -> Pad to enable 7x7 chunks  
                layer_img_width = 4;
                layer_img_height = 4;
                layer_num_channels = 32;
                layer_patch_size = 7;  // Use 7x7 chunks with padding
                layer_stride = 1;      // Single extraction from padded coordinate (0,0)
                layer_use_full_image = 1;  // Extract one 7x7 chunk from padded 4x4
                // Use model's default asymmetric padding: 4x4 -> 7x7 (perfect fit!)
                layer_pad_top = 1; layer_pad_bottom = 2;
                layer_pad_left = 1; layer_pad_right = 2;
            end
            // Removed layer_idx=4 (2x2x64) as it's not needed
            default: begin // Default to input layer
                layer_img_width = 32;
                layer_img_height = 32;
                layer_num_channels = 1;
                layer_patch_size = 7;
                layer_stride = 4;
                layer_use_full_image = 0;
                layer_pad_top = 0; layer_pad_bottom = 0; layer_pad_left = 0; layer_pad_right = 0;
            end
        endcase
        
        // Calculate padded dimensions
        /* verilator lint_off WIDTHEXPAND */
        /* verilator lint_off WIDTHTRUNC */
        padded_img_width = ($clog2(MAX_IMG_W + 3 + 1)+1)'(layer_img_width + layer_pad_left + layer_pad_right);
        padded_img_height = ($clog2(MAX_IMG_H + 3 + 1)+1)'(layer_img_height + layer_pad_top + layer_pad_bottom);
        /* verilator lint_on WIDTHTRUNC */
        /* verilator lint_on WIDTHEXPAND */
    end

    // Calculate total channel groups needed (always 4 channels per group)
    logic [$clog2(MAX_CHANNELS/4)-1:0] total_channel_groups;
    /* verilator lint_off WIDTHEXPAND */
    assign total_channel_groups = ($clog2(MAX_CHANNELS/4))'((layer_num_channels + 3) / 4);
    /* verilator lint_on WIDTHEXPAND */
    
    // Calculate spatial positions per RAM read based on channel count
    logic [$clog2(17)-1:0] spatial_positions_per_read;
    always_comb begin
        case (layer_num_channels)
            1:  spatial_positions_per_read = 16;  // 16 spatial positions, 1 channel each
            2:  spatial_positions_per_read = 8;   // 8 spatial positions, 2 channels each  
            4:  spatial_positions_per_read = 4;   // 4 spatial positions, 4 channels each
            8:  spatial_positions_per_read = 2;   // 2 spatial positions, 8 channels each
            16: spatial_positions_per_read = 1;   // 1 spatial position, 16 channels
            32: spatial_positions_per_read = 1;   // 1 spatial position, 16 channels (need 2 reads)
            64: spatial_positions_per_read = 1;   // 1 spatial position, 16 channels (need 4 reads)
            default: spatial_positions_per_read = 1;
        endcase
    end

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        LOADING_BLOCK,
        BLOCK_READY,
        WAIT_NEXT_SPATIAL,
        COMPLETE
    } state_t;
    state_t state;

    // Current spatial block position (in padded coordinate space)
    logic signed [$clog2(MAX_IMG_W + 3)-1:0] block_start_col;
    logic signed [$clog2(MAX_IMG_H + 3)-1:0] block_start_row;
    
    // Current channel group tracking
    logic [$clog2(MAX_CHANNELS/4)-1:0] channel_group;
    
    // Buffer to store current patch
    logic [31:0] buffer_patch [6:0][6:0];
    
    // RAM data cache
    logic [127:0] ram_data_cache;
    logic ram_data_valid;
    
    // Loading counters
    logic [$clog2(64)-1:0] ram_read_count;
    logic [$clog2(64)-1:0] total_ram_reads;
    logic [$clog2(7)-1:0] extract_row, extract_col;
    
    // Calculate total RAM reads needed based on layer-specific patch size
    logic [$clog2(64)-1:0] total_positions;
    always_comb begin
        total_positions = ($clog2(64))'(layer_patch_size * layer_patch_size);
        // For layers with many channels (32, 64), we need multiple reads per spatial position
        /* verilator lint_off WIDTHEXPAND */
        if (layer_num_channels > 16) begin
            total_ram_reads = ($clog2(64))'(total_positions * ((layer_num_channels + 15) / 16));  // Round up
        end else begin
            total_ram_reads = ($clog2(64))'((total_positions + spatial_positions_per_read - 1) / spatial_positions_per_read);
        end
        /* verilator lint_on WIDTHEXPAND */
    end
    
    // State machine
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            block_start_row <= 0;
            block_start_col <= 0;
            channel_group <= 0;
            ram_read_count <= 0;
            ram_data_valid <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start_extraction) begin
                        // Start from top-left of padded space (negative coordinates for top/left padding)
                        block_start_row <= -($clog2(MAX_IMG_H + 3))'(layer_pad_top);
                        block_start_col <= -($clog2(MAX_IMG_W + 3))'(layer_pad_left);
                        channel_group <= 0;
                        ram_read_count <= 0;
                        ram_data_valid <= 0;
                        state <= LOADING_BLOCK;
                    end
                end
                
                LOADING_BLOCK: begin
                    if (ram_data_valid) begin
                        if (ram_read_count >= total_ram_reads - 1) begin
                            state <= BLOCK_READY;
                        end else begin
                            ram_read_count <= ram_read_count + 1;
                            ram_data_valid <= 1'b0;
                        end
                    end
                end
                
                BLOCK_READY: begin
                    if (next_channel_group) begin
                        if (channel_group >= total_channel_groups - 1) begin
                            // All channels done for this spatial block 
                            channel_group <= 0;
                            state <= WAIT_NEXT_SPATIAL;
                        end else begin
                            // Move to next channel group for same spatial block
                            channel_group <= channel_group + 1;
                            ram_read_count <= 0;
                            ram_data_valid <= 0;
                            state <= LOADING_BLOCK;
                        end
                    end
                end
                
                WAIT_NEXT_SPATIAL: begin
                    if (next_spatial_block) begin
                        // Calculate next spatial position based on stride and current position
                        if (layer_use_full_image) begin
                            // For full image extraction, we're done after one extraction
                            state <= COMPLETE;
                        end else begin
                            // Normal stride-based advancement
                            /* verilator lint_off WIDTHEXPAND */
                            if (($clog2(MAX_IMG_W + 3 + 1))'(block_start_col + layer_stride + layer_patch_size) > padded_img_width) begin
                                // Move to next row
                                block_start_col <= -($clog2(MAX_IMG_W + 3))'(layer_pad_left);
                                if (($clog2(MAX_IMG_H + 3 + 1))'(block_start_row + layer_stride + layer_patch_size) > padded_img_height) begin
                                    // Extraction complete
                                    state <= COMPLETE;
                                end else begin
                                    block_start_row <= block_start_row + ($clog2(MAX_IMG_H + 3))'(layer_stride);
                                    ram_read_count <= 0;
                                    ram_data_valid <= 0;
                                    state <= LOADING_BLOCK;
                                end
                            end else begin
                                // Move to next column
                                block_start_col <= block_start_col + ($clog2(MAX_IMG_W + 3))'(layer_stride);
                                ram_read_count <= 0;
                                ram_data_valid <= 0;
                                state <= LOADING_BLOCK;
                            end
                            /* verilator lint_on WIDTHEXPAND */
                        end
                    end
                end
                
                COMPLETE: begin
                    // Stay in complete state until reset
                    state <= COMPLETE;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    // Address calculation variables (moved outside always_comb)
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] calculated_addr;
    logic is_padding_region;
    logic [$clog2(64)-1:0] linear_position;
    logic [$clog2(7)-1:0] buf_row, buf_col;
    logic signed [$clog2(MAX_IMG_W + 3)-1:0] mem_row, mem_col;
    logic [$clog2(MAX_IMG_W)-1:0] actual_mem_row, actual_mem_col;
    logic [$clog2(64)-1:0] channel_read_idx;
    logic signed [$clog2(MAX_IMG_W + 3)-1:0] actual_img_row, actual_img_col;
    logic [$clog2(64)-1:0] spatial_idx;  // Moved to module level to avoid latch

    // Address calculation
    always_comb begin
        calculated_addr = 0;
        is_padding_region = 1'b0;
        linear_position = 0;
        buf_row = 0;
        buf_col = 0;
        mem_row = 0;
        mem_col = 0;
        actual_mem_row = 0;
        actual_mem_col = 0;
        channel_read_idx = 0;
        actual_img_row = 0;
        actual_img_col = 0;
        spatial_idx = 0;  // Always assign to avoid latch
        
        if (state == LOADING_BLOCK) begin
            if (layer_num_channels > 16) begin
                // High channel count - handle differently (32+ channels need multiple reads)
                /* verilator lint_off WIDTHEXPAND */
                spatial_idx = ($clog2(64))'(ram_read_count / ((layer_num_channels + 15) / 16));
                channel_read_idx = ($clog2(64))'(ram_read_count % ((layer_num_channels + 15) / 16));
                /* verilator lint_on WIDTHEXPAND */
                buf_row = ($clog2(7))'(spatial_idx / layer_patch_size);
                buf_col = ($clog2(7))'(spatial_idx % layer_patch_size);
                
                mem_row = block_start_row + ($clog2(MAX_IMG_W + 3))'(buf_row);
                mem_col = block_start_col + ($clog2(MAX_IMG_W + 3))'(buf_col);
                
                // Check if this position is within the actual image (not padding)
                // Convert from padded coordinates to actual image coordinates
                actual_img_row = mem_row - ($clog2(MAX_IMG_W + 3))'(layer_pad_top);
                actual_img_col = mem_col - ($clog2(MAX_IMG_W + 3))'(layer_pad_left);
                
                if (actual_img_row >= 0 && actual_img_row < layer_img_height && 
                    actual_img_col >= 0 && actual_img_col < layer_img_width) begin
                    actual_mem_row = actual_img_row[$clog2(MAX_IMG_W)-1:0];
                    actual_mem_col = actual_img_col[$clog2(MAX_IMG_W)-1:0];
                    /* verilator lint_off WIDTHEXPAND */
                    calculated_addr = ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4))'(
                        (actual_mem_row * layer_img_width + actual_mem_col) * 
                        ((layer_num_channels + 15) / 16) + channel_read_idx);
                    /* verilator lint_on WIDTHEXPAND */
                    is_padding_region = 1'b0;
                end else begin
                    is_padding_region = 1'b1;
                end
            end else begin
                // Normal case - single or few reads per spatial position
                linear_position = ($clog2(64))'(ram_read_count * spatial_positions_per_read);
                buf_row = ($clog2(7))'(linear_position / layer_patch_size);
                buf_col = ($clog2(7))'(linear_position % layer_patch_size);
                
                mem_row = block_start_row + ($clog2(MAX_IMG_W + 3))'(buf_row);
                mem_col = block_start_col + ($clog2(MAX_IMG_W + 3))'(buf_col);
                
                // Check if this position is within the actual image (not padding)
                // Convert from padded coordinates to actual image coordinates
                actual_img_row = mem_row - ($clog2(MAX_IMG_W + 3))'(layer_pad_top);
                actual_img_col = mem_col - ($clog2(MAX_IMG_W + 3))'(layer_pad_left);
                
                if (actual_img_row >= 0 && actual_img_row < layer_img_height && 
                    actual_img_col >= 0 && actual_img_col < layer_img_width) begin
                    actual_mem_row = actual_img_row[$clog2(MAX_IMG_W)-1:0];
                    actual_mem_col = actual_img_col[$clog2(MAX_IMG_W)-1:0];
                    calculated_addr = ($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4))'(
                        actual_mem_row * layer_img_width + actual_mem_col);
                    is_padding_region = 1'b0;
                end else begin
                    is_padding_region = 1'b1;
                end
            end
        end
    end
    
    // RAM control - don't read from memory for padding regions
    assign ram_re = (state == LOADING_BLOCK && !ram_data_valid && !is_padding_region);
    assign ram_addr = calculated_addr;
    
    // RAM data caching
    always_ff @(posedge clk) begin
        if (reset) begin
            ram_data_cache <= '0;
            ram_data_valid <= 1'b0;
        end else begin
            if (ram_re) begin
                // Normal memory read
                ram_data_cache <= {ram_dout0, ram_dout1, ram_dout2, ram_dout3};
                ram_data_valid <= 1'b1;
            end else if (state == LOADING_BLOCK && !ram_data_valid && is_padding_region) begin
                // Padding region - provide zeros
                ram_data_cache <= '0;
                ram_data_valid <= 1'b1;
            end else if (state == IDLE || state == WAIT_NEXT_SPATIAL || 
                        (state == BLOCK_READY && next_channel_group && channel_group < total_channel_groups - 1)) begin
                ram_data_valid <= 1'b0;
            end
        end
    end
    
    // Data extraction and buffer management - Extract 4 consecutive channels
    function automatic [31:0] extract_channel_data(
        input [127:0] cached_data
        ,input [$clog2(17)-1:0] spatial_pos_index
        ,input [$clog2(MAX_CHANNELS/4)-1:0] ch_group
        ,input [$clog2(17)-1:0] positions_per_read
        ,input [$clog2(MAX_CHANNELS+1)-1:0] num_ch
    );
        logic [31:0] result;
        logic [6:0] base_channel;  // Increased width to handle MAX_CHANNELS/4 * 4
        logic [7:0] ch0, ch1, ch2, ch3;  // 4 channels to extract
        
        result = 32'h00000000;
        base_channel = 7'(ch_group * 4);  // Channel group 0->chs 0-3, group 1->chs 4-7, etc.
        
        // Extract 4 consecutive channels based on memory layout
        case (positions_per_read)
            16: begin // 1 channel per spatial position (16 spatial positions in 128-bit read)
                // For 1 channel layers, only channel 0 exists, others are padded with zero
                if (base_channel == 0 && num_ch >= 1) begin
                    ch0 = cached_data[(spatial_pos_index * 8) +: 8];
                end else ch0 = 8'h00;
                ch1 = 8'h00;  // Pad with zeros
                ch2 = 8'h00;
                ch3 = 8'h00;
                result = {ch3, ch2, ch1, ch0};
            end
            
            8: begin // 2 channels per spatial position (8 spatial positions in 128-bit read)
                // Extract channels based on group
                if (base_channel == 0 && num_ch >= 2) begin
                    ch0 = cached_data[(spatial_pos_index * 16) + 0 +: 8];
                    ch1 = cached_data[(spatial_pos_index * 16) + 8 +: 8];
                end else begin
                    ch0 = 8'h00;
                    ch1 = 8'h00;
                end
                ch2 = 8'h00;  // Pad remaining with zeros
                ch3 = 8'h00;
                result = {ch3, ch2, ch1, ch0};
            end
            
            4: begin // 4 channels per spatial position (4 spatial positions in 128-bit read)
                // Perfect fit - extract all 4 channels
                if (base_channel == 0 && num_ch >= 4) begin
                    ch0 = cached_data[(spatial_pos_index * 32) + 0 +: 8];
                    ch1 = cached_data[(spatial_pos_index * 32) + 8 +: 8];
                    ch2 = cached_data[(spatial_pos_index * 32) + 16 +: 8];
                    ch3 = cached_data[(spatial_pos_index * 32) + 24 +: 8];
                end else begin
                    ch0 = 8'h00; ch1 = 8'h00; ch2 = 8'h00; ch3 = 8'h00;
                end
                result = {ch3, ch2, ch1, ch0};
            end
            
            2: begin // 8 channels per spatial position (2 spatial positions in 128-bit read)
                // Extract 4 channels from the 8 available, based on channel group
                if ((7'(base_channel) + 3) < 7'(num_ch)) begin
                    ch0 = cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 0 +: 8];
                    ch1 = cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 8 +: 8];
                    ch2 = cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 16 +: 8];
                    ch3 = cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 24 +: 8];
                end else begin
                    // Handle partial groups with padding
                    ch0 = ((7'(base_channel) + 0) < 7'(num_ch)) ? cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 0 +: 8] : 8'h00;
                    ch1 = ((7'(base_channel) + 1) < 7'(num_ch)) ? cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 8 +: 8] : 8'h00;
                    ch2 = ((7'(base_channel) + 2) < 7'(num_ch)) ? cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 16 +: 8] : 8'h00;
                    ch3 = ((7'(base_channel) + 3) < 7'(num_ch)) ? cached_data[(spatial_pos_index * 64) + (base_channel * 8) + 24 +: 8] : 8'h00;
                end
                result = {ch3, ch2, ch1, ch0};
            end
            
            1: begin // 16+ channels per spatial position (1 spatial position in 128-bit read)
                // Extract 4 channels from the 16 available, based on channel group
                if ((7'(base_channel) + 3) < 7'(num_ch)) begin
                    ch0 = cached_data[(base_channel * 8) + 0 +: 8];
                    ch1 = cached_data[(base_channel * 8) + 8 +: 8];
                    ch2 = cached_data[(base_channel * 8) + 16 +: 8];
                    ch3 = cached_data[(base_channel * 8) + 24 +: 8];
                end else begin
                    // Handle partial groups with padding
                    ch0 = ((7'(base_channel) + 0) < 7'(num_ch)) ? cached_data[(base_channel * 8) + 0 +: 8] : 8'h00;
                    ch1 = ((7'(base_channel) + 1) < 7'(num_ch)) ? cached_data[(base_channel * 8) + 8 +: 8] : 8'h00;
                    ch2 = ((7'(base_channel) + 2) < 7'(num_ch)) ? cached_data[(base_channel * 8) + 16 +: 8] : 8'h00;
                    ch3 = ((7'(base_channel) + 3) < 7'(num_ch)) ? cached_data[(base_channel * 8) + 24 +: 8] : 8'h00;
                end
                result = {ch3, ch2, ch1, ch0};
            end
            
            default: begin
                result = 32'h00000000;
            end
        endcase
        
        return result;
    endfunction
    
    // Store extracted data into patch buffer
    always_ff @(posedge clk) begin
        if (ram_data_valid && state == LOADING_BLOCK) begin
            // Extract spatial positions based on layer configuration
            for (int pos = 0; pos < spatial_positions_per_read; pos++) begin
                logic [$clog2(7)-1:0] buf_row_local, buf_col_local;
                logic [$clog2(64)-1:0] abs_pos;
                logic [$clog2(64)-1:0] spatial_idx_local;
                
                if (layer_num_channels > 16) begin
                    // High channel count - handle differently (32+ channels need multiple reads)
                    /* verilator lint_off WIDTHEXPAND */
                    spatial_idx_local = ($clog2(64))'(ram_read_count / ((layer_num_channels + 15) / 16));
                    /* verilator lint_on WIDTHEXPAND */
                    buf_row_local = ($clog2(7))'(spatial_idx_local / layer_patch_size);
                    buf_col_local = ($clog2(7))'(spatial_idx_local % layer_patch_size);
                    
                    if (buf_row_local < layer_patch_size && buf_col_local < layer_patch_size) begin
                        buffer_patch[buf_row_local][buf_col_local] <= extract_channel_data(
                            ram_data_cache, 0, channel_group, 1, layer_num_channels);
                    end
                end else begin
                    // Normal case - extract for each spatial position in this RAM read
                    abs_pos = ($clog2(64))'(ram_read_count * spatial_positions_per_read + pos);
                    buf_row_local = ($clog2(7))'(abs_pos / layer_patch_size);
                    buf_col_local = ($clog2(7))'(abs_pos % layer_patch_size);
                    
                    if (buf_row_local < layer_patch_size && buf_col_local < layer_patch_size) begin
                        buffer_patch[buf_row_local][buf_col_local] <= extract_channel_data(
                            ram_data_cache, pos[$clog2(17)-1:0], channel_group, 
                            spatial_positions_per_read, layer_num_channels);
                    end
                end
            end
        end
    end
    
    // Output assignments
    assign patch_data_out = buffer_patch;
    assign patches_valid = (state == BLOCK_READY) && ram_data_valid;
    assign block_ready = (state == BLOCK_READY) && ram_data_valid;
    assign extraction_complete = (state == COMPLETE);
    assign all_channels_done = (state == WAIT_NEXT_SPATIAL);
    assign buffer_loading_complete = ram_data_valid;
    
    // Address outputs (convert from padded coordinates)
    assign block_start_col_addr = (block_start_col >= -($clog2(MAX_IMG_W + 3))'(layer_pad_left)) ? 
                                  ($clog2(MAX_IMG_W))'(block_start_col + ($clog2(MAX_IMG_W + 3))'(layer_pad_left)) : '0;
    assign block_start_row_addr = (block_start_row >= -($clog2(MAX_IMG_H + 3))'(layer_pad_top)) ? 
                                  ($clog2(MAX_IMG_H))'(block_start_row + ($clog2(MAX_IMG_H + 3))'(layer_pad_top)) : '0;
    assign block_coords_valid = (block_start_col >= -($clog2(MAX_IMG_W + 3))'(layer_pad_left)) && 
                               (block_start_row >= -($clog2(MAX_IMG_H + 3))'(layer_pad_top)) && 
                               (($clog2(MAX_IMG_W + 3 + 1))'(block_start_col) < (padded_img_width - ($clog2(MAX_IMG_W + 3 + 1))'(layer_patch_size))) && 
                               (($clog2(MAX_IMG_H + 3 + 1))'(block_start_row) < (padded_img_height - ($clog2(MAX_IMG_H + 3 + 1))'(layer_patch_size)));

endmodule 