`include "sys_types.svh"

module patch_extractor #(
    parameter int MAX_IMG_W = 64,
    parameter int MAX_IMG_H = 64,
    parameter int MAX_CHANNELS = 64,
    parameter int PATCH_SIZE = 4,
    parameter int STRIDE = 1
) (
    input  logic                            clk,
    input  logic                            reset,
    
    // Configuration for current layer
    input  logic [$clog2(MAX_IMG_W+1)-1:0] img_width,
    input  logic [$clog2(MAX_IMG_H+1)-1:0] img_height,
    input  logic [$clog2(MAX_CHANNELS+1)-1:0] num_channels,
    
    // Control signals
    input  logic                            start_patch_extraction,
    input  logic                            next_patch,  // Move to next channel, or next position if all channels done
    output logic                            patch_ready,
    output logic                            extraction_complete,
    
    // Status outputs
    output logic [$clog2(MAX_CHANNELS+1)-1:0] current_channel,
    output logic [$clog2(MAX_IMG_W)-1:0]      current_patch_col,
    output logic [$clog2(MAX_IMG_H)-1:0]      current_patch_row,
    output logic                               all_channels_done_for_position,
    
    // Interface to tensor_ram (8-bit granular addressing for channel-last layout)
    output logic                            ram_re,
    output logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16)-1:0] ram_addr,
    input  logic [31:0]                     ram_dout0,
    input  logic [31:0]                     ram_dout1,
    input  logic [31:0]                     ram_dout2,
    input  logic [31:0]                     ram_dout3,
    
    // Interface to sliding_window
    output logic [31:0]                     patch_A0_out,
    output logic [31:0]                     patch_A1_out,
    output logic [31:0]                     patch_A2_out,
    output logic [31:0]                     patch_A3_out,
    output logic                            patch_valid
);

    // State machine states
    typedef enum logic [3:0] {
        IDLE,
        FETCH_PIXELS,
        WAIT_RAM_DATA,     // New state to wait for RAM data
        ASSEMBLE_PATCH,
        PATCH_AVAILABLE,
        ADVANCE_CHANNEL,
        ADVANCE_POSITION
    } state_t;
    
    state_t current_state, next_state;
    
    // Position tracking - channels iterate first for each position
    logic [$clog2(MAX_IMG_W)-1:0]      patch_col;
    logic [$clog2(MAX_IMG_H)-1:0]      patch_row;
    logic [$clog2(MAX_CHANNELS+1)-1:0] patch_channel;
    logic [1:0]                         current_patch_row_offset;
    logic [1:0]                         pixels_fetched_in_row; // Changed to 2 bits for 0-3 range
    
    // Storage for the 4x4 patch pixels for current channel
    logic [7:0] patch_pixels [0:PATCH_SIZE-1][0:PATCH_SIZE-1];
    logic [PATCH_SIZE-1:0] patch_rows_complete;
    
    // Assembled 4x4 patch (4 rows of 4 pixels each, packed as 32-bit words)
    logic [31:0] assembled_patch [0:3];
    
    // Address calculation helpers - properly sized
    logic [$clog2(MAX_IMG_W*MAX_CHANNELS+1)-1:0] pixels_per_row;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-1:0] pixel_base_addr;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16+1)-1:0] word_addr;
    logic [3:0] pixel_offset_in_word;
    
    // Calculate pixels per row in memory (width * channels)
    assign pixels_per_row = img_width * num_channels;
    
    // Output current position info
    assign current_channel = patch_channel;
    assign current_patch_col = patch_col;
    assign current_patch_row = patch_row;
    assign all_channels_done_for_position = (patch_channel == num_channels - 1);
    
    // Address calculation for channel-last layout: [row, col, channel]
    // Base address for current patch pixel being fetched
    always_comb begin
        // Calculate pixel address: (patch_row + row_offset) * width * channels + (patch_col + col_offset) * channels + channel
        pixel_base_addr = ({{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-$clog2(MAX_IMG_H)){1'b0}}, patch_row} + 
                          {{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-2){1'b0}}, current_patch_row_offset}) * 
                          {{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-$clog2(MAX_IMG_W*MAX_CHANNELS+1)){1'b0}}, pixels_per_row} + 
                         ({{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-$clog2(MAX_IMG_W)){1'b0}}, patch_col} + 
                          {{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-2){1'b0}}, pixels_fetched_in_row}) * 
                          {{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-$clog2(MAX_CHANNELS+1)){1'b0}}, num_channels} + 
                          {{($clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-$clog2(MAX_CHANNELS+1)){1'b0}}, patch_channel};
        
        // Convert to 128-bit word address (16 pixels per word)
        /* verilator lint_off WIDTHTRUNC */
        word_addr = pixel_base_addr >> 4;  // Divide by 16
        /* verilator lint_on WIDTHTRUNC */
        pixel_offset_in_word = pixel_base_addr[3:0]; // Modulo 16
    end
    
    // Register the pixel offset to match RAM data latency
    logic [3:0] pixel_offset_delayed;
    always_ff @(posedge clk) begin
        if (reset) begin
            pixel_offset_delayed <= 4'b0;
        end else if (ram_re) begin
            pixel_offset_delayed <= pixel_offset_in_word;
        end
    end
    
    // Extract current pixel from the 128-bit word based on delayed offset
    logic [7:0] current_pixel;
    always_comb begin
        case (pixel_offset_delayed)
            4'd0:  current_pixel = ram_dout3[31:24];  // MSB of fourth 32-bit word (leftmost in hex)
            4'd1:  current_pixel = ram_dout3[23:16];
            4'd2:  current_pixel = ram_dout3[15:8];
            4'd3:  current_pixel = ram_dout3[7:0];    // LSB of fourth 32-bit word
            4'd4:  current_pixel = ram_dout2[31:24];  // MSB of third 32-bit word  
            4'd5:  current_pixel = ram_dout2[23:16];
            4'd6:  current_pixel = ram_dout2[15:8];
            4'd7:  current_pixel = ram_dout2[7:0];    // LSB of third 32-bit word
            4'd8:  current_pixel = ram_dout1[31:24];  // MSB of second 32-bit word
            4'd9:  current_pixel = ram_dout1[23:16];
            4'd10: current_pixel = ram_dout1[15:8];
            4'd11: current_pixel = ram_dout1[7:0];    // LSB of second 32-bit word
            4'd12: current_pixel = ram_dout0[31:24];  // MSB of first 32-bit word
            4'd13: current_pixel = ram_dout0[23:16];
            4'd14: current_pixel = ram_dout0[15:8];
            4'd15: current_pixel = ram_dout0[7:0];    // LSB of first 32-bit word (rightmost in hex)
            default: current_pixel = 8'h0;
        endcase
    end
    
    // State machine
    always_ff @(posedge clk) begin
        if (reset) begin
            current_state <= IDLE;
            patch_col <= '0;
            patch_row <= '0;
            patch_channel <= '0;
            current_patch_row_offset <= 2'b0;
            pixels_fetched_in_row <= 2'b0;
            patch_rows_complete <= 4'b0;
            for (int i = 0; i < PATCH_SIZE; i++) begin
                for (int j = 0; j < PATCH_SIZE; j++) begin
                    patch_pixels[i][j] <= 8'h0;
                end
                assembled_patch[i] <= 32'h0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    if (start_patch_extraction) begin
                        patch_col <= '0;
                        patch_row <= '0;
                        patch_channel <= '0;
                        current_patch_row_offset <= 2'b0;
                        pixels_fetched_in_row <= 2'b0;
                        patch_rows_complete <= 4'b0;
                    end
                end
                
                FETCH_PIXELS: begin
                    // Just initiate the RAM read, data will be ready next cycle
                end
                
                WAIT_RAM_DATA: begin
                    // Store the fetched pixel in the patch array (data is now ready)
                    patch_pixels[current_patch_row_offset][pixels_fetched_in_row] <= current_pixel;
                    
                    if (pixels_fetched_in_row < 2'd3) begin // PATCH_SIZE - 1 = 3
                        // More pixels needed in this row
                        pixels_fetched_in_row <= pixels_fetched_in_row + 2'b1;
                    end else begin
                        // Row complete, mark it and move to next row
                        pixels_fetched_in_row <= 2'b0;
                        patch_rows_complete[current_patch_row_offset] <= 1'b1;
                        
                        if (current_patch_row_offset < 2'd3) begin // PATCH_SIZE - 1 = 3
                            current_patch_row_offset <= current_patch_row_offset + 2'b1;
                        end else begin
                            // All rows fetched
                            current_patch_row_offset <= 2'b0;
                        end
                    end
                end
                
                ASSEMBLE_PATCH: begin
                    // Pack the 4x4 pixels into four 32-bit words (little-endian order)
                    for (int row = 0; row < PATCH_SIZE; row++) begin
                        assembled_patch[row] <= {patch_pixels[row][0], patch_pixels[row][1], 
                                               patch_pixels[row][2], patch_pixels[row][3]};
                    end
                end
                
                PATCH_AVAILABLE: begin
                    if (next_patch) begin
                        patch_rows_complete <= 4'b0;
                        // Clear patch storage for next extraction
                        for (int i = 0; i < PATCH_SIZE; i++) begin
                            for (int j = 0; j < PATCH_SIZE; j++) begin
                                patch_pixels[i][j] <= 8'h0;
                            end
                        end
                    end
                end
                
                ADVANCE_CHANNEL: begin
                    if (patch_channel < num_channels - 1) begin
                        // Move to next channel, same position
                        patch_channel <= patch_channel + 1;
                    end else begin
                        // All channels done for this position, reset to channel 0
                        patch_channel <= '0;
                    end
                    current_patch_row_offset <= 2'b0;
                    pixels_fetched_in_row <= 2'b0;
                end
                
                ADVANCE_POSITION: begin
                    // Move to next spatial position with proper width handling
                    /* verilator lint_off WIDTHEXPAND */
                    if ((patch_col + STRIDE) < (img_width - PATCH_SIZE + 1)) begin
                        patch_col <= patch_col + STRIDE[$clog2(MAX_IMG_W)-1:0];
                    end else begin
                        patch_col <= '0;
                        if ((patch_row + STRIDE) < (img_height - PATCH_SIZE + 1)) begin
                            patch_row <= patch_row + STRIDE[$clog2(MAX_IMG_H)-1:0];
                        end
                        // Note: don't reset position if we're at the end - let next_state go to IDLE
                    end
                    /* verilator lint_on WIDTHEXPAND */
                    current_patch_row_offset <= 2'b0;
                    pixels_fetched_in_row <= 2'b0;
                end
                
                default: begin
                    // Should not reach here
                    current_state <= IDLE;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if (start_patch_extraction) next_state = FETCH_PIXELS;
            end
            
            FETCH_PIXELS: begin
                // After initiating RAM read, wait for data
                next_state = WAIT_RAM_DATA;
            end
            
            WAIT_RAM_DATA: begin
                // After storing pixel, check if we need more pixels
                if ((current_patch_row_offset == 2'd3) && (pixels_fetched_in_row == 2'd3)) begin
                    next_state = ASSEMBLE_PATCH;
                end else begin
                    next_state = FETCH_PIXELS;  // Fetch next pixel
                end
            end
            
            ASSEMBLE_PATCH: begin
                next_state = PATCH_AVAILABLE;
            end
            
            PATCH_AVAILABLE: begin
                if (next_patch) next_state = ADVANCE_CHANNEL;
            end
            
            ADVANCE_CHANNEL: begin
                if (patch_channel < num_channels - 1) begin
                    // More channels to process for this position
                    next_state = FETCH_PIXELS;
                end else begin
                    // All channels done for this position, advance position
                    next_state = ADVANCE_POSITION;
                end
            end
            
            ADVANCE_POSITION: begin
                /* verilator lint_off WIDTHEXPAND */
                // Check if we would advance beyond the valid range
                if ((patch_col + STRIDE) >= (img_width - PATCH_SIZE + 1)) begin
                    // Moving to next row
                    if ((patch_row + STRIDE) >= (img_height - PATCH_SIZE + 1)) begin
                        next_state = IDLE; // No more rows, extraction complete
                    end else begin
                        next_state = FETCH_PIXELS; // Go to next row
                    end
                end else begin
                    next_state = FETCH_PIXELS; // Continue on same row
                end
                /* verilator lint_on WIDTHEXPAND */
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // RAM control signals
    assign ram_re = (current_state == FETCH_PIXELS);
    assign ram_addr = word_addr[$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16)-1:0];
    
    // Output to sliding window
    assign patch_A0_out = assembled_patch[0];
    assign patch_A1_out = assembled_patch[1];
    assign patch_A2_out = assembled_patch[2];
    assign patch_A3_out = assembled_patch[3];
    assign patch_valid = (current_state == PATCH_AVAILABLE);
    assign patch_ready = (current_state == PATCH_AVAILABLE);
    
    // Extraction complete signal - when we've done all positions and are back to IDLE
    /* verilator lint_off WIDTHEXPAND */
    assign extraction_complete = (current_state == ADVANCE_POSITION) && 
                                ((patch_col + STRIDE) >= (img_width - PATCH_SIZE + 1)) &&
                                ((patch_row + STRIDE) >= (img_height - PATCH_SIZE + 1));
    /* verilator lint_on WIDTHEXPAND */

endmodule
