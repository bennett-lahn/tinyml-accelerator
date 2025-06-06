module unif_buf_0 #(
    parameter MAX_IMG_W = 64,
    parameter MAX_IMG_H = 64,
    parameter MAX_CHANNELS = 64,
    parameter MAX_PADDING = 3
)(
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_extraction,
    input logic next_channel_group,
    input logic next_spatial_block,
    
    // Memory interface
    output logic ram_re,
    output logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] ram_addr,
    input logic [31:0] ram_dout0, ram_dout1, ram_dout2, ram_dout3,
    input logic ram_data_valid,
    
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

    // Layer 0 specific configuration - optimized for single channel, small images
    localparam LAYER_0_IMG_WIDTH = 32;
    localparam LAYER_0_IMG_HEIGHT = 32; 
    localparam LAYER_0_NUM_CHANNELS = 1;
    localparam LAYER_0_PATCH_SIZE = 7;  // 7x7 patches for layer 0
    localparam LAYER_0_STRIDE = 4;      // Stride of 4 for layer 0
    localparam LAYER_0_PAD_TOP = 1;
    localparam LAYER_0_PAD_BOTTOM = 2;
    localparam LAYER_0_PAD_LEFT = 1;
    localparam LAYER_0_PAD_RIGHT = 2;
    
    // Padded dimensions
    localparam PADDED_WIDTH = LAYER_0_IMG_WIDTH + LAYER_0_PAD_LEFT + LAYER_0_PAD_RIGHT;   // 38
    localparam PADDED_HEIGHT = LAYER_0_IMG_HEIGHT + LAYER_0_PAD_TOP + LAYER_0_PAD_BOTTOM; // 38

    typedef logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/4)-1:0] addr_t;
    typedef addr_t addr_array_t [0:6][0:6];

    typedef logic [5:0] offset_t;
    typedef offset_t offset_array_t [0:6][0:6];

    typedef logic all_padding_t [0:6][0:6];
    // State machine
    typedef enum logic [2:0] {
        IDLE,
        LOADING_BLOCK,
        BLOCK_READY_STATE,
        COMPLETE
    } state_t;
    state_t state;

    // Current spatial block position (in padded coordinate space)
    logic signed [2*$clog2(PADDED_WIDTH)-1:0] block_start_col;
    logic signed [2*$clog2(PADDED_HEIGHT)-1:0] block_start_row;
    
    // Current spatial block position (in unpadded/memory coordinate space)
    logic [$clog2(LAYER_0_IMG_WIDTH)-1:0] block_start_col_unpadded;
    logic [$clog2(LAYER_0_IMG_HEIGHT)-1:0] block_start_row_unpadded;
    
    // Buffer to store current 7x7 patch (single channel, so 1 byte per position)
    logic [31:0] buffer_patch [6:0][6:0];
    
    // RAM data cache for 128-bit reads
    logic [127:0] ram_data_cache;
    logic cache_valid;
    
    // Loading counters
    logic [$clog2(64)-1:0] ram_read_count;
    logic [$clog2(64)-1:0] total_ram_reads;
    logic [$clog2(7)-1:0] extract_row, extract_col;
    logic is_padding;

    addr_array_t mem_addrs;
    offset_array_t mem_offsets;
    all_padding_t all_padding;

    logic last_addr;
    logic [2:0] buf_row, buf_col;
    logic [2:0] patch_row, patch_col;


    //for every position these have all the addresses and offsets for the 7x7 patch
    assign mem_addrs = calc_addr(block_start_col, block_start_row);
    assign mem_offsets = calc_offset(block_start_col, block_start_row);
    assign all_padding = is_row_in_padding(block_start_col, block_start_row);

    
    // For single channel layer 0: need one read per row of 7x7 patch
    // A 7x7 patch requires reading from 7 different rows in row-major layout
    assign total_ram_reads = 7;

    function automatic addr_array_t calc_addr(
        input logic signed [2*$clog2(PADDED_WIDTH)-1:0] col,
        input logic signed [2*$clog2(PADDED_HEIGHT)-1:0] row,
    );
        // Local array to accumulate results
       addr_array_t blocks;
        // idx ranges up to (31*32 + 31) = 1023 → needs 10 bits
        logic [2*$clog2(PADDED_WIDTH*PADDED_HEIGHT)-1:0] idx;

        begin
            for (int i = 0; i < 7; i++) begin
                for (int j = 0; j < 7; j++) begin
                    // Compute linear index of pixel (r+i, c+j):
                    //   idx = (r + i)*32 + (c + j)
                    // Since 32 = (1 << 5), we can do (r + i) << 5:
                    
                    // Compute 1‐based block number = floor(idx/16) + 1
                    //   (idx >> 4) is floor(idx/16).  Result is in 6 bits (0..63).
                    // Assign into 8 bits (zero‐extended).
                    if (row+i < 0 || row+i >= 32 || col+j < 0 || col+j >= 32) begin
                        blocks[i][j] = '0;
                    end else begin
                        idx = ((row + i) << 5) + (col + j);
                        blocks[i][j] = (idx >> 4) ;
                    end
                end
            end
            return blocks;
        end
    endfunction

    function automatic offset_array_t calc_offset(
        input logic signed [2*$clog2(PADDED_WIDTH)-1:0] col,
        input logic signed [2*$clog2(PADDED_HEIGHT)-1:0] row,
    );
        offset_array_t offsets;
        logic [$clog2(PADDED_WIDTH*PADDED_HEIGHT)-1:0] idx;
        // Local array to accumulate result
        begin
            for (int i = 0; i < 7; i++) begin
                for (int j = 0; j < 7; j++) begin
                    // Compute linear index of pixel (r+i, c+j):
                    if (row+i < 0 || row+i >= 32 || col+j < 0 || col+j >= 32) begin
                        offsets[i][j] = '0;
                    end else begin
                        idx = ((row + i) << 5) + (col + j);
                        // Compute offset = idx % 16 → idx[3:0]
                        // Zero‐extend to 8 bits by concatenating four zero bits above:
                        offsets[i][j] = {4'b0000, idx[3:0]};
                    end
                end
            end
            return offsets;
        end
    endfunction
        
        
    
    // Check if current row is in padding region
    function automatic all_padding_t is_row_in_padding(
        input logic signed [2*$clog2(PADDED_WIDTH)-1:0] col,
        input logic signed [2*$clog2(PADDED_HEIGHT)-1:0] row,

    );
        all_padding_t padding;
        for (int i = 0; i < 7; i++) begin
            for (int j = 0; j < 7; j++) begin
                padding[i][j] = (row+i < 0 || row+i >= 32 || col+j < 0 || col+j >= 32);
            end
        end
        return padding;
    endfunction
    
   
    // State machine
    always_ff @(posedge clk) begin
        state <= state;
        // $display("state: %d", state);
        if (reset) begin
            state <= IDLE;
            block_start_col <= -LAYER_0_PAD_LEFT;
            block_start_row <= -LAYER_0_PAD_TOP;
            ram_read_count <= 0;
            extract_row <= 0;
            extract_col <= 0;
            cache_valid <= 1'b0;
            ram_data_cache <= '0;
            last_addr <= 1'b0;
            for (int i = 0; i < 7; i++) begin
                for (int j = 0; j < 7; j++) begin
                    buffer_patch[i][j] <= '0;
                end
            end
        end else begin

            case (state)
                IDLE: begin
                    if (start_extraction) begin
                        state <= LOADING_BLOCK;
                        ram_read_count <= 0;
                        extract_row <= 0;
                        extract_col <= 0;
                        cache_valid <= 1'b0;
                        buf_row <= 0;
                        buf_col <= 0;
                        last_addr <= 1'b0;
                        // Clear buffer
                        for (int i = 0; i < 7; i++) begin
                            for (int j = 0; j < 7; j++) begin
                                buffer_patch[i][j] <= '0;
                            end
                        end
                    end
                end
                
                LOADING_BLOCK: begin
                    // $display("buffer_start_col: %d, buffer_start_row: %d", block_start_col, block_start_row);
                        if (buf_col == 6) begin
                            buf_col <= 0;
                            if (buf_row == 6) begin
                                buf_row <= 0;
                                last_addr <= 1'b1;
                            end else begin
                                buf_row <= buf_row + 1;
                            end
                        end else begin
                            buf_col <= buf_col + 1;
                        end
                        if(last_addr) begin
                            $display("buffer_start_col: %d, buffer_start_row: %d", block_start_col, block_start_row);
                            $display("last_addr, moving to block_ready_state");
                            state <= BLOCK_READY_STATE;
                        end
                        patch_row <= buf_row;
                        patch_col <= buf_col;
                end
                BLOCK_READY_STATE: begin
                    if (next_spatial_block) begin
                        $display("next_spatial_block, moving to LOADING_BLOCK");
                        // Move to next spatial block
                        state <= LOADING_BLOCK;
                        if (block_start_col + LAYER_0_STRIDE < (LAYER_0_IMG_WIDTH + LAYER_0_PAD_RIGHT - LAYER_0_PATCH_SIZE + 1)) begin
                            //move right
                            $display("moving right");
                            block_start_col <= block_start_col + LAYER_0_STRIDE;
                            // $display("mystery addition: %d", block_start_col + LAYER_0_STRIDE);
                        end else if (block_start_row + LAYER_0_STRIDE < (LAYER_0_IMG_HEIGHT + LAYER_0_PAD_BOTTOM - LAYER_0_PATCH_SIZE + 1)) begin
                            //move down
                            $display("moving down");
                            block_start_row <= block_start_row + LAYER_0_STRIDE;
                            block_start_col <= -LAYER_0_PAD_LEFT;
                        end else begin
                            $display("no more blocks to process");
                            //no more blocks to process
                            state <= COMPLETE;
                        end
                    end
                        
                    
                    ram_read_count <= 0;
                    cache_valid <= 1'b0;
                    buf_row <= 0;
                    buf_col <= 0;
                    last_addr <= 1'b0;
                end     
                COMPLETE: begin
                    if (start_extraction) begin
                        // Reset for new extraction
                        state <= IDLE;
                        block_start_col <= -LAYER_0_PAD_LEFT;
                        block_start_row <= -LAYER_0_PAD_TOP;
                        last_addr <= 1'b0;
                        buf_row <= 0;
                        buf_col <= 0;
                    end
                end
                default: begin
                    state <= IDLE;
                    block_start_col <= -LAYER_0_PAD_LEFT;
                    block_start_row <= -LAYER_0_PAD_TOP;
                    last_addr <= 1'b0;
                    buf_row <= 0;
                    buf_col <= 0;
                end
            endcase
        end
    end

    logic [127:0] ram_data;
    // capturing ram data
    always_comb begin
        if(ram_data_valid) begin
            ram_data = {ram_dout0, ram_dout1, ram_dout2, ram_dout3};
        end
        else begin
            ram_data = '0;
        end
    end

    always_ff @(posedge clk) begin
        if(state == LOADING_BLOCK) begin
            if(!all_padding[patch_row][patch_col]) begin
                if(ram_data_valid) begin
                    // $display("ram_data: %h", ram_data);
                    // $display("mem_offsets: %d", ram_data[(15-mem_offsets[patch_row][patch_col])*8 +: 8]);
                    // $display("patch_row: %d, patch_col: %d", patch_row, patch_col);
                    buffer_patch[patch_row][patch_col] <= {24'b0, ram_data[(15-mem_offsets[patch_row][patch_col])*8 +: 8]};
                end
            end
            else begin
                // $display("padding %d %d", patch_row, patch_col);
                buffer_patch[patch_row][patch_col] <= '0;
            end
        end
    end

    
   


    // RAM control
    always_comb begin
        ram_re = (state == LOADING_BLOCK); //the padded value always reads 0,0 and will be ignored
        ram_addr = mem_addrs[buf_row][buf_col];
    end
    
    // Output assignments
    assign patch_data_out = buffer_patch;
    assign patches_valid = (state == BLOCK_READY_STATE);
    assign block_ready = (state == BLOCK_READY_STATE);
    assign extraction_complete = (state == COMPLETE);
    assign all_channels_done = (state == COMPLETE);
    assign buffer_loading_complete = cache_valid && (state == BLOCK_READY_STATE);
    
    // Address outputs (convert from padded coordinates to image coordinates)
    assign block_start_col_addr = (block_start_col >= -LAYER_0_PAD_LEFT) ? 
                                  (block_start_col + LAYER_0_PAD_LEFT) : 0;
    assign block_start_row_addr = (block_start_row >= -LAYER_0_PAD_TOP) ? 
                                  (block_start_row + LAYER_0_PAD_TOP) : 0;
    assign block_coords_valid = (state == BLOCK_READY_STATE);
    
endmodule

