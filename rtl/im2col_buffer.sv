/**
 * Implements the im2col algorithm for a single-channel image with dynamic IMG_W and IMG_H.
 * Reads an image from memory (row-major), computes all kernel patches,
 * and stores them sequentially into an internal RAM that acts as a FIFO.
 * Outputs a 128-bit patch from this FIFO when a read is requested.
 * Assumes no padding. Kernel is fixed at 4x4, Stride is a fixed parameter.
 */
module im2col #(
    parameter MAX_IMG_DIM = 256,        // Maximum possible image width or height
    parameter KERNEL_W = 4,             // Kernel width (fixed at 4)
    parameter KERNEL_H = 4,             // Kernel height (fixed at 4)
    parameter STRIDE_W = 1,             // Stride along width (fixed parameter)
    parameter STRIDE_H = 1,             // Stride along height (fixed parameter)
    parameter DATA_WIDTH = 8,           // Bit width of each pixel
    // MEM_ADDR_WIDTH sized for maximum possible single-channel image
    parameter MEM_ADDR_WIDTH = $clog2(MAX_IMG_DIM * MAX_IMG_DIM),
    // MAX_TOTAL_PATCHES for sizing the internal RAM
    localparam MAX_THEORETICAL_OUT_DIM_FOR_PATCH_RAM = (MAX_IMG_DIM - KERNEL_W) / STRIDE_W + 1, // Simplified for sizing
    parameter MAX_TOTAL_PATCHES = MAX_THEORETICAL_OUT_DIM_FOR_PATCH_RAM * MAX_THEORETICAL_OUT_DIM_FOR_PATCH_RAM
) (
    input logic clk,
    input logic reset,

    // Configuration Inputs
    input logic [$clog2(MAX_IMG_DIM)-1:0] current_img_w_in,
    input logic [$clog2(MAX_IMG_DIM)-1:0] current_img_h_in,

    input logic start_im2col,           // Start the im2col process
    output logic im2col_busy,           // Module is busy processing/filling internal RAM
    output logic im2col_done,           // All patches for current config are stored in internal RAM/FIFO

    // Memory Read Interface for Input Image
    output logic [MEM_ADDR_WIDTH-1:0] mem_addr,
    output logic mem_rd_en,
    input  logic [DATA_WIDTH-1:0] mem_data_in,
    input  logic mem_data_valid,

    // FIFO Read Interface for the internal patches RAM
    input  logic patch_fifo_read_en_in,         // Request to read the next patch from FIFO
    output logic [127:0] patch_fifo_data_out,   // Data from the FIFO (a full 128-bit patch)
    output logic patch_fifo_valid_out,          // Indicates patch_fifo_data_out is valid for one cycle
    output logic patch_fifo_empty_out           // Indicates the FIFO has no more patches to read
);

    localparam PATCH_NUM_ELEMENTS = KERNEL_H * KERNEL_W; // 16
    localparam PATCH_TOTAL_BITS = PATCH_NUM_ELEMENTS * DATA_WIDTH; // 128

    // Internal registers for latched configuration
    logic [$clog2(MAX_IMG_DIM)-1:0] r_img_w;
    logic [$clog2(MAX_IMG_DIM)-1:0] r_img_h;
    logic [$clog2(MAX_IMG_DIM)-1:0] r_out_h; // Calculated output height
    logic [$clog2(MAX_IMG_DIM)-1:0] r_out_w; // Calculated output width
    logic [$clog2(MAX_TOTAL_PATCHES)-1:0] total_patches_for_current_config_reg;

    // Internal RAM to store all generated patches (FIFO storage)
    logic [127:0] internal_all_patches_ram [0:MAX_TOTAL_PATCHES-1];
    logic [$clog2(MAX_TOTAL_PATCHES)-1:0] current_ram_write_idx; // Write pointer for the RAM
    logic [$clog2(MAX_TOTAL_PATCHES)-1:0] current_ram_read_idx;  // Read pointer for the RAM/FIFO

    // FIFO control signals
    logic [$clog2(MAX_TOTAL_PATCHES+1)-1:0] fifo_count; // Number of patches currently in FIFO
    logic r_patch_fifo_valid_out;                       // Registered valid output for FIFO read
    logic [127:0] r_patch_fifo_data_out;                // Registered data output for FIFO read

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        CALC_OUT_DIMS,
        INIT_PATCH_FETCH_COUNTERS,
        CALC_ADDR,
        WAIT_MEM
    } state_t;
    state_t current_state, next_state;

    // Counters for iterating through output patches (where the kernel top-left is placed)
    logic [$clog2(MAX_IMG_DIM)-1:0] out_patch_row_idx;
    logic [$clog2(MAX_IMG_DIM)-1:0] out_patch_col_idx;

    // Counters for iterating within the current kernel patch being fetched
    logic [$clog2(KERNEL_H)-1:0] kernel_row_offset;
    logic [$clog2(KERNEL_W)-1:0] kernel_col_offset;
    logic [$clog2(PATCH_NUM_ELEMENTS)-1:0] current_patch_element_count;

    logic [DATA_WIDTH-1:0] internal_patch_buffer [0:PATCH_NUM_ELEMENTS-1]; // Temp buffer for one patch
    logic mem_read_issued_flag;
    logic r_im2col_done_reg; // Registered version of im2col_done

    logic write_to_ram_strobe; // Combinational signal to strobe RAM write
    logic can_write_to_fifo;
    logic can_read_from_fifo;

    logic [$clog2(MAX_IMG_DIM)-1:0] current_pixel_img_row_abs;
    logic [$clog2(MAX_IMG_DIM)-1:0] current_pixel_img_col_abs;
    logic [$clog2(MAX_TOTAL_PATCHES*2)-1:0] temp_total_patches;

    assign temp_total_patches = r_out_h * r_out_w;

    assign can_write_to_fifo = write_to_ram_strobe;
    assign can_read_from_fifo  = patch_fifo_read_en_in && (fifo_count > 0);

    // --- Sequential Logic ---
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            current_state <= IDLE;
            r_img_w <= '0; r_img_h <= '0; r_out_h <= '0; r_out_w <= '0;
            total_patches_for_current_config_reg <= '0;
            out_patch_row_idx <= '0; out_patch_col_idx <= '0;
            current_ram_write_idx <= '0;
            kernel_row_offset <= '0; kernel_col_offset <= '0; current_patch_element_count <= '0;
            mem_read_issued_flag <= 1'b0;
            r_im2col_done_reg <= 1'b0;

            // FIFO specific resets
            current_ram_read_idx <= '0;
            fifo_count <= '0;
            r_patch_fifo_valid_out <= 1'b0;
            r_patch_fifo_data_out <= '0;
        end else begin
            current_state <= next_state;
            r_patch_fifo_valid_out <= 1'b0; // Default to low, asserted on successful read

            // Configuration Latching and Done Flag Management
            if (current_state == IDLE && next_state == CALC_OUT_DIMS) begin
                r_im2col_done_reg <= 1'b0; // Clear done on new start
                r_img_w <= current_img_w_in;
                r_img_h <= current_img_h_in;
                current_ram_write_idx <= '0;
                current_ram_read_idx <= '0; // Reset FIFO read pointer on new task
                fifo_count <= '0;           // Reset FIFO count on new task
                out_patch_row_idx <= '0;
                out_patch_col_idx <= '0;
            end

            if (current_state == CALC_OUT_DIMS && next_state == INIT_PATCH_FETCH_COUNTERS) begin
                if (r_img_h >= KERNEL_H) r_out_h <= (r_img_h - KERNEL_H) / STRIDE_H + 1; else r_out_h <= '0;
                if (r_img_w >= KERNEL_W) r_out_w <= (r_img_w - KERNEL_W) / STRIDE_W + 1; else r_out_w <= '0;
                
                // automatic logic [$clog2(MAX_TOTAL_PATCHES*2)-1:0] temp_total_patches;
                if (r_out_h == 0 || r_out_w == 0) begin
                    total_patches_for_current_config_reg <= '0;
                end else begin
                    // temp_total_patches = ;
                    total_patches_for_current_config_reg <= (temp_total_patches > MAX_TOTAL_PATCHES) ? MAX_TOTAL_PATCHES : temp_total_patches[$clog2(MAX_TOTAL_PATCHES)-1:0];
                end
            end
            
            if (next_state == INIT_PATCH_FETCH_COUNTERS && 
                (current_state == CALC_OUT_DIMS || (current_state == WAIT_MEM && write_to_ram_strobe))) begin
                kernel_row_offset <= '0; kernel_col_offset <= '0; current_patch_element_count <= '0;
            end
            
            if (current_state == CALC_ADDR && mem_rd_en) mem_read_issued_flag <= 1'b1;

            if (current_state == WAIT_MEM && mem_data_valid && mem_read_issued_flag) begin
                internal_patch_buffer[current_patch_element_count] <= mem_data_in;
                mem_read_issued_flag <= 1'b0; 
                if (current_patch_element_count != PATCH_NUM_ELEMENTS - 1) begin
                    current_patch_element_count <= current_patch_element_count + 1;
                    if (kernel_col_offset == 2'(KERNEL_W - 1)) begin 
                        kernel_col_offset <= '0; 
                        kernel_row_offset <= kernel_row_offset + 1;
                    end else begin
                        kernel_col_offset <= kernel_col_offset + 1;
                    end
                end
            end

            // Write to internal RAM (FIFO storage)
            if (write_to_ram_strobe) begin
                logic [(PATCH_TOTAL_BITS)-1:0] data_to_write;
                for (int i_pack_ff = 0; i_pack_ff < PATCH_NUM_ELEMENTS; i_pack_ff = i_pack_ff + 1) begin
                    data_to_write[(i_pack_ff * DATA_WIDTH) +: DATA_WIDTH] = internal_patch_buffer[i_pack_ff];
                end
                // Assign to RAM (handle potential width mismatch, though PATCH_TOTAL_BITS is 128)
                // if (PATCH_TOTAL_BITS == 128) 
                internal_all_patches_ram[current_ram_write_idx] <= data_to_write;
                // else if (PATCH_TOTAL_BITS < 128) internal_all_patches_ram[current_ram_write_idx] <= { (128 - PATCH_TOTAL_BITS){1'b0}, data_to_write };
                // else internal_all_patches_ram[current_ram_write_idx] <= data_to_write[127:0];
                
                // current_ram_write_idx is incremented in FIFO control section if write occurs
            end
            
            // FIFO Read and Write Control
           

            

            if (can_read_from_fifo) begin
                r_patch_fifo_data_out <= internal_all_patches_ram[current_ram_read_idx];
                current_ram_read_idx <= current_ram_read_idx + 1;
                r_patch_fifo_valid_out <= 1'b1;
            end

            // Update fifo_count based on operations
            if (can_write_to_fifo && !can_read_from_fifo) begin // Write only
                fifo_count <= fifo_count + 1;
                current_ram_write_idx <= current_ram_write_idx + 1; // Advance write pointer
            end else if (!can_write_to_fifo && can_read_from_fifo) begin // Read only
                fifo_count <= fifo_count - 1;
            end else if (can_write_to_fifo && can_read_from_fifo) begin // Read and Write
                // fifo_count remains the same
                current_ram_write_idx <= current_ram_write_idx + 1; // Advance write pointer
            end

            // Advance global patch iterators after a patch is written to RAM
            if (write_to_ram_strobe) begin
                 if (out_patch_col_idx == r_out_w - 1) begin
                    out_patch_col_idx <= '0;
                    if (out_patch_row_idx != r_out_h - 1) begin
                         out_patch_row_idx <= out_patch_row_idx + 1;
                    end
                end else begin
                    out_patch_col_idx <= out_patch_col_idx + 1;
                end
            end

            // Set im2col_done flag
            // Done when write pointer reaches the total number of patches to be generated for this config.
            if ( (total_patches_for_current_config_reg > 0 && current_ram_write_idx == total_patches_for_current_config_reg && !can_write_to_fifo) || // All written
                 (current_state == CALC_OUT_DIMS && next_state == INIT_PATCH_FETCH_COUNTERS && total_patches_for_current_config_reg == 0 && (r_out_h > 0 || r_out_w > 0) ) // Config results in 0 patches
                ) begin
                r_im2col_done_reg <= 1'b1;
            end
        end
    end

    // --- Combinational Logic ---
    assign write_to_ram_strobe = (current_state == WAIT_MEM && mem_data_valid && mem_read_issued_flag && (current_patch_element_count == PATCH_NUM_ELEMENTS - 1));

    assign patch_fifo_data_out = r_patch_fifo_data_out;
    assign patch_fifo_valid_out = r_patch_fifo_valid_out;
    assign patch_fifo_empty_out = (fifo_count == 0);

    assign im2col_done = r_im2col_done_reg;
    assign im2col_busy = (current_state != IDLE) && !r_im2col_done_reg;

    always_comb begin
        next_state = current_state;
        mem_addr = 'x;
        mem_rd_en = 1'b0;

        

        current_pixel_img_row_abs = out_patch_row_idx * STRIDE_H + {{6{1'b0}}, kernel_row_offset};
        current_pixel_img_col_abs = out_patch_col_idx * STRIDE_W + {{6{1'b0}}, kernel_col_offset};

        case (current_state)
            IDLE: begin
                if (start_im2col) next_state = CALC_OUT_DIMS;
            end
            CALC_OUT_DIMS: begin
                next_state = INIT_PATCH_FETCH_COUNTERS;
            end
            INIT_PATCH_FETCH_COUNTERS: begin
                if ( (total_patches_for_current_config_reg == 0 && (r_out_h > 0 || r_out_w > 0)) || // Config resulted in 0 patches
                     (current_ram_write_idx >= total_patches_for_current_config_reg && total_patches_for_current_config_reg > 0) // All patches written
                   ) begin
                    next_state = IDLE; // Done with im2col processing for this image
                end else begin
                    next_state = CALC_ADDR;
                end
            end
            CALC_ADDR: begin
                if (current_pixel_img_row_abs < r_img_h && current_pixel_img_col_abs < r_img_w) begin
                    mem_addr = current_pixel_img_row_abs * r_img_w + {{8{1'b0}}, current_pixel_img_col_abs};
                    mem_rd_en = 1'b1;
                end
                next_state = WAIT_MEM; 
            end
            WAIT_MEM: begin
                if (mem_data_valid && mem_read_issued_flag) begin
                    if (current_patch_element_count == PATCH_NUM_ELEMENTS - 1) begin // Patch complete
                        // write_to_ram_strobe will be high. RAM write, fifo_count, and write_ptr update in FF.
                        // Global patch iterators (out_patch_row/col_idx) also update in FF based on strobe.
                        next_state = INIT_PATCH_FETCH_COUNTERS; // Prepare for next patch or determine completion
                    end else begin // More elements needed for the current patch
                        next_state = CALC_ADDR;
                    end
                end
                // Else, stay in WAIT_MEM if waiting for data or if an error occurred in CALC_ADDR
            end
            default: next_state = IDLE;
        endcase
    end
endmodule
