module unif_buf_1 #(
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
    output logic all_channels_done,
    
    // Address calculation outputs
    output logic [$clog2(MAX_IMG_W)-1:0] block_start_col_addr,
    output logic [$clog2(MAX_IMG_H)-1:0] block_start_row_addr,
    
    // Data outputs - variable patch sizes based on layer
    output logic [31:0] patch_data_out [6:0][6:0],  // Maximum 7x7 buffer
    output logic patches_valid
);

    // Layer 1 specific configuration - optimized for single channel, small images
    localparam LAYER_1_IMG_WIDTH = 16;
    localparam LAYER_1_IMG_HEIGHT = 16; 
    localparam LAYER_1_NUM_CHANNELS = 8;
    localparam LAYER_1_PATCH_SIZE = 7;  // 7x7 patches for layer 0
    localparam LAYER_1_STRIDE = 4;      // Stride of 4 for layer 0
    localparam LAYER_1_PAD_TOP = 1;
    localparam LAYER_1_PAD_BOTTOM = 2;
    localparam LAYER_1_PAD_LEFT = 1;
    localparam LAYER_1_PAD_RIGHT = 2;
    
    // Padded dimensions
    localparam PADDED_WIDTH = LAYER_1_IMG_WIDTH + LAYER_1_PAD_LEFT + LAYER_1_PAD_RIGHT;   // 38
    localparam PADDED_HEIGHT = LAYER_1_IMG_HEIGHT + LAYER_1_PAD_TOP + LAYER_1_PAD_BOTTOM; // 38

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
    logic [2:0] block_start_ch; //either 0 or 4 for this layer
    
    // Current spatial block position (in unpadded/memory coordinate space)
    logic [$clog2(LAYER_1_IMG_WIDTH)-1:0] block_start_col_unpadded;
    logic [$clog2(LAYER_1_IMG_HEIGHT)-1:0] block_start_row_unpadded;
    
    // Buffer to store current 7x7 patch (single channel, so 1 byte per position)
    logic [31:0] buffer_patch [6:0][6:0];
    
    // RAM data cache for 128-bit reads
    logic [127:0] ram_data_cache;
    
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
    assign mem_addrs = calc_addr(block_start_col, block_start_row, block_start_ch);
    assign mem_offsets = calc_offset(block_start_col, block_start_row, block_start_ch);
    assign all_padding = is_row_in_padding(block_start_col, block_start_row, block_start_ch);

    
    // For single channel layer 0: need one read per row of 7x7 patch
    // A 7x7 patch requires reading from 7 different rows in row-major layout
    assign total_ram_reads = 7;

    function automatic addr_array_t calc_addr(
        input logic signed [2*$clog2(PADDED_WIDTH)-1:0] col,
        input logic signed [2*$clog2(PADDED_HEIGHT)-1:0] row,
        input logic        [2:0] ch
    );
        // Local array to accumulate results
        addr_array_t blocks;
        logic signed [2*$clog2(PADDED_WIDTH)-1:0]  row_idx;
        logic signed [2*$clog2(PADDED_HEIGHT)-1:0] col_idx;
        logic        [2:0]  chan_idx;
        logic        [10:0] idx;

        begin
            for (int i = 0; i < 7; i++) begin
                for (int j = 0; j < 7; j++) begin
                    row_idx = row + i;
                    col_idx = col + j;

                    // If spatial coordinate is out‐of‐bounds, mark as padding:
                    if (row_idx+i < 0 || row_idx+i >= 16 || col_idx+j < 0 || col_idx+j >= 16) begin
                        blocks[i][j] = 8'd0;
                    end else begin
                        // “ch” must be either 0 or 4, so (ch..ch+3) ∈ [0..7].
                        // Compute idx for the first channel in this group:
                        //   idx = (((row_idx * 16) + col_idx) * 8) + ch
                        idx = ((((row_idx << 4) + col_idx) << 3) + ch);
                        // 0‐based block = floor(idx/16)
                        blocks[i][j] = idx >> 4;
                    end
                end
            end
            return blocks;
        end
    endfunction

    function automatic offset_array_t calc_offset(
        input logic signed [2*$clog2(PADDED_WIDTH)-1:0] col,
        input logic signed [2*$clog2(PADDED_HEIGHT)-1:0] row,
        input logic        [2:0] ch
    );
        offset_array_t offsets;
        logic signed [2*$clog2(PADDED_WIDTH)-1:0]  row_idx;
        logic signed [2*$clog2(PADDED_HEIGHT)-1:0] col_idx;
        logic        [2:0]  chan_idx;
        logic        [10:0] idx;
        logic        [3:0]  raw_off;

        begin
            for (int i = 0; i < 7; i++) begin
                for (int j = 0; j < 7; j++) begin
                    row_idx = row + i;
                    col_idx = col + j;

                    if (row_idx+i < 0 || row_idx+i >= 16 || col_idx+j < 0 || col_idx+j >= 16) begin
                        offsets[i][j] = 8'd0;
                    end else begin
                        idx = ((((row_idx << 4) + col_idx) << 3) + ch);
                    raw_off = idx[3:0];              // idx % 16
                        offsets[i][j] = {4'b0000, raw_off};
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
        input logic        [2:0] ch
    );
        all_padding_t padding;
        for (int i = 0; i < 7; i++) begin
            for (int j = 0; j < 7; j++) begin
                padding[i][j] = (row+i < 0 || row+i >= 16 || col+j < 0 || col+j >= 16);
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
            block_start_col <= -LAYER_1_PAD_LEFT;
            block_start_row <= -LAYER_1_PAD_TOP;
            block_start_ch <= 0;
            ram_read_count <= 0;
            extract_row <= 0;
            extract_col <= 0;
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
                        buf_row <= 0;
                        buf_col <= 0;
                        last_addr <= 1'b0;
                        block_start_ch <= 0;
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
                            // $display("buffer_start_col: %d, buffer_start_row: %d", block_start_col, block_start_row);
                            // $display("last_addr, moving to block_ready_state");
                            state <= BLOCK_READY_STATE;
                        end
                        patch_row <= buf_row;
                        patch_col <= buf_col;
                end
                BLOCK_READY_STATE: begin
                    if (next_spatial_block) begin
                        // $display("next_spatial_block, moving to LOADING_BLOCK");
                        // Move to next spatial block
                        state <= LOADING_BLOCK;
                        if (block_start_ch == 4) begin
                            if (block_start_col + LAYER_1_STRIDE < (LAYER_1_IMG_WIDTH + LAYER_1_PAD_RIGHT - LAYER_1_PATCH_SIZE + 1)) begin
                                //move right
                                // $display("moving right");
                                block_start_col <= block_start_col + LAYER_1_STRIDE;
                                block_start_ch <= 0;
                            // $display("mystery addition: %d", block_start_col + LAYER_0_STRIDE);
                        end else if (block_start_row + LAYER_1_STRIDE < (LAYER_1_IMG_HEIGHT + LAYER_1_PAD_BOTTOM - LAYER_1_PATCH_SIZE + 1)) begin
                            //move down
                            // $display("moving down");
                            block_start_row <= block_start_row + LAYER_1_STRIDE;
                            block_start_ch <= 0;
                            block_start_col <= -LAYER_1_PAD_LEFT;
                        end else begin
                                // $display("no more blocks to process");
                                //no more blocks to process
                                state <= COMPLETE;
                            end
                        end
                        else begin
                            //move inwards
                            block_start_ch <= 4;
                        end
                    end
                        
                    
                    ram_read_count <= 0;
                    buf_row <= 0;
                    buf_col <= 0;
                    last_addr <= 1'b0;
                end     
                COMPLETE: begin
                    if (start_extraction) begin
                        // Reset for new extraction
                        state <= IDLE;
                        block_start_col <= -LAYER_1_PAD_LEFT;
                        block_start_row <= -LAYER_1_PAD_TOP;
                        block_start_ch <= 0;
                        last_addr <= 1'b0;
                        buf_row <= 0;
                        buf_col <= 0;
                    end
                end
                default: begin
                    state <= IDLE;
                    block_start_col <= -LAYER_1_PAD_LEFT;
                    block_start_row <= -LAYER_1_PAD_TOP;
                    block_start_ch <= 0;
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
                    buffer_patch[patch_row][patch_col] <= {
                        ram_data[ ((15 - mem_offsets[patch_row][patch_col]) * 8 - 24) +: 8 ],  // offset+3 → starts at base_bit - 3*8
                        ram_data[ ((15 - mem_offsets[patch_row][patch_col]) * 8 - 16) +: 8 ],  // offset+2 → starts at base_bit - 2*8
                        ram_data[ ((15 - mem_offsets[patch_row][patch_col]) * 8 -  8) +: 8 ],  // offset+1 → starts at base_bit - 1*8
                        ram_data[ ((15 - mem_offsets[patch_row][patch_col]) * 8 -  0) +: 8 ]    // offset   → starts at base_bit
                    };
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
    assign all_channels_done = (state == COMPLETE);
    
    // Address outputs (convert from padded coordinates to image coordinates)
    assign block_start_col_addr = (block_start_col >= -LAYER_1_PAD_LEFT) ? 
                                  (block_start_col + LAYER_1_PAD_LEFT) : 0;
    assign block_start_row_addr = (block_start_row >= -LAYER_1_PAD_TOP) ? 
                                  (block_start_row + LAYER_1_PAD_TOP) : 0;
    
endmodule

