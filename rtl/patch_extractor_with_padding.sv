`include "sys_types.svh"

module patch_extractor_with_padding #(
    parameter int MAX_IMG_W = 64,
    parameter int MAX_IMG_H = 64,
    parameter int MAX_CHANNELS = 64,
    parameter int PATCH_SIZE = 4,
    parameter int STRIDE = 1,
    // Zero padding parameters for 4x4 SAME padding
    parameter int PAD_LEFT = 1,
    parameter int PAD_RIGHT = 2,
    parameter int PAD_TOP = 1,
    parameter int PAD_BOTTOM = 2
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
        WAIT_RAM_DATA,     
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
    logic [1:0]                         pixels_fetched_in_row;
    
    // Storage for the 4x4 patch pixels for current channel
    logic [7:0] patch_pixels [0:PATCH_SIZE-1][0:PATCH_SIZE-1];
    logic [PATCH_SIZE-1:0] patch_rows_complete;
    
    // Assembled 4x4 patch (4 rows of 4 pixels each, packed as 32-bit words)
    logic [31:0] assembled_patch [0:3];
    
    // Address calculation helpers
    logic [$clog2(MAX_IMG_W*MAX_CHANNELS+1)-1:0] pixels_per_row;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS+1)-1:0] pixel_base_addr;
    logic [$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16+1)-1:0] word_addr;
    logic [3:0] pixel_offset_in_word;
    
    // Padding calculation - current absolute pixel coordinates
    logic signed [$clog2(MAX_IMG_W+PAD_LEFT+PAD_RIGHT+1)-1:0] abs_pixel_row, abs_pixel_col;
    logic is_padding_pixel;
    logic is_valid_ram_access;
    
    // Calculate pixels per row in memory (width * channels)
    assign pixels_per_row = img_width * num_channels;
    
    // Output current position info
    assign current_channel = patch_channel;
    assign current_patch_col = patch_col;
    assign current_patch_row = patch_row;
    assign all_channels_done_for_position = (patch_channel == num_channels - 1);
    
    // Calculate absolute pixel coordinates (including padding offset)
    always_comb begin
        // Simplified arithmetic for absolute pixel coordinates
        abs_pixel_row = $signed({1'b0, patch_row}) + $signed({1'b0, current_patch_row_offset}) - $signed({1'b0, PAD_TOP});
        abs_pixel_col = $signed({1'b0, patch_col}) + $signed({1'b0, pixels_fetched_in_row}) - $signed({1'b0, PAD_LEFT});
        
        // Check if current pixel is padding (out of bounds) or valid image data
        is_padding_pixel = (abs_pixel_row < 0) || 
                          (abs_pixel_row >= $signed({1'b0, img_height})) ||
                          (abs_pixel_col < 0) || 
                          (abs_pixel_col >= $signed({1'b0, img_width}));
        
        is_valid_ram_access = !is_padding_pixel;
    end
    
    // Address calculation for channel-last layout: [row, col, channel]
    // Only calculate when accessing valid (non-padding) pixels
    always_comb begin
        if (is_valid_ram_access) begin
            // Calculate pixel address for valid pixels using simpler arithmetic
            pixel_base_addr = $unsigned(abs_pixel_row) * pixels_per_row + 
                             $unsigned(abs_pixel_col) * num_channels + 
                             patch_channel;
            
            // Convert to 128-bit word address (16 pixels per word)
            word_addr = pixel_base_addr >> 4;  // Divide by 16
            pixel_offset_in_word = pixel_base_addr[3:0]; // Modulo 16
        end else begin
            // For padding pixels, don't access RAM
            pixel_base_addr = '0;
            word_addr = '0;
            pixel_offset_in_word = '0;
        end
    end
    
    // Register the pixel offset and padding flag to match RAM data latency
    logic [3:0] pixel_offset_delayed;
    logic is_padding_delayed;
    always_ff @(posedge clk) begin
        if (reset) begin
            pixel_offset_delayed <= 4'b0;
            is_padding_delayed <= 1'b0;
        end else if (ram_re || is_padding_pixel) begin
            pixel_offset_delayed <= pixel_offset_in_word;
            is_padding_delayed <= is_padding_pixel;
        end
    end
    
    // Extract current pixel from the 128-bit word or return zero for padding
    logic [7:0] current_pixel;
    always_comb begin
        if (is_padding_delayed) begin
            current_pixel = 8'h00;  // Zero padding
        end else begin
            case (pixel_offset_delayed)
                4'd0:  current_pixel = ram_dout3[31:24];
                4'd1:  current_pixel = ram_dout3[23:16];
                4'd2:  current_pixel = ram_dout3[15:8];
                4'd3:  current_pixel = ram_dout3[7:0];
                4'd4:  current_pixel = ram_dout2[31:24];
                4'd5:  current_pixel = ram_dout2[23:16];
                4'd6:  current_pixel = ram_dout2[15:8];
                4'd7:  current_pixel = ram_dout2[7:0];
                4'd8:  current_pixel = ram_dout1[31:24];
                4'd9:  current_pixel = ram_dout1[23:16];
                4'd10: current_pixel = ram_dout1[15:8];
                4'd11: current_pixel = ram_dout1[7:0];
                4'd12: current_pixel = ram_dout0[31:24];
                4'd13: current_pixel = ram_dout0[23:16];
                4'd14: current_pixel = ram_dout0[15:8];
                4'd15: current_pixel = ram_dout0[7:0];
                default: current_pixel = 8'h0;
            endcase
        end
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
                    // Initiate RAM read only if accessing valid (non-padding) pixel
                end
                
                WAIT_RAM_DATA: begin
                    // Store the fetched pixel (zero for padding, actual data for valid pixels)
                    patch_pixels[current_patch_row_offset][pixels_fetched_in_row] <= current_pixel;
                    
                    if (pixels_fetched_in_row < 2'd3) begin
                        pixels_fetched_in_row <= pixels_fetched_in_row + 2'b1;
                    end else begin
                        pixels_fetched_in_row <= 2'b0;
                        patch_rows_complete[current_patch_row_offset] <= 1'b1;
                        
                        if (current_patch_row_offset < 2'd3) begin
                            current_patch_row_offset <= current_patch_row_offset + 2'b1;
                        end else begin
                            current_patch_row_offset <= 2'b0;
                        end
                    end
                end
                
                ASSEMBLE_PATCH: begin
                    // Pack the 4x4 pixels into four 32-bit words
                    for (int row = 0; row < PATCH_SIZE; row++) begin
                        assembled_patch[row] <= {patch_pixels[row][0], patch_pixels[row][1], 
                                               patch_pixels[row][2], patch_pixels[row][3]};
                    end
                end
                
                PATCH_AVAILABLE: begin
                    if (next_patch) begin
                        patch_rows_complete <= 4'b0;
                        for (int i = 0; i < PATCH_SIZE; i++) begin
                            for (int j = 0; j < PATCH_SIZE; j++) begin
                                patch_pixels[i][j] <= 8'h0;
                            end
                        end
                    end
                end
                
                ADVANCE_CHANNEL: begin
                    if (patch_channel < num_channels - 1) begin
                        patch_channel <= patch_channel + 1;
                    end else begin
                        patch_channel <= '0;
                    end
                    current_patch_row_offset <= 2'b0;
                    pixels_fetched_in_row <= 2'b0;
                end
                
                ADVANCE_POSITION: begin
                    // With padding, output size equals input size for SAME padding
                    if ((patch_col + STRIDE) < img_width) begin
                        patch_col <= patch_col + STRIDE[$clog2(MAX_IMG_W)-1:0];
                    end else begin
                        patch_col <= '0;
                        if ((patch_row + STRIDE) < img_height) begin
                            patch_row <= patch_row + STRIDE[$clog2(MAX_IMG_H)-1:0];
                        end
                    end
                    current_patch_row_offset <= 2'b0;
                    pixels_fetched_in_row <= 2'b0;
                end
                
                default: begin
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
                next_state = WAIT_RAM_DATA;
            end
            
            WAIT_RAM_DATA: begin
                if ((current_patch_row_offset == 2'd3) && (pixels_fetched_in_row == 2'd3)) begin
                    next_state = ASSEMBLE_PATCH;
                end else begin
                    next_state = FETCH_PIXELS;
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
                    next_state = FETCH_PIXELS;
                end else begin
                    next_state = ADVANCE_POSITION;
                end
            end
            
            ADVANCE_POSITION: begin
                // Check if we've processed all positions (SAME padding: output size = input size)
                if ((patch_col + STRIDE) >= img_width) begin
                    if ((patch_row + STRIDE) >= img_height) begin
                        next_state = IDLE; // All positions complete
                    end else begin
                        next_state = FETCH_PIXELS; // Next row
                    end
                end else begin
                    next_state = FETCH_PIXELS; // Continue on same row
                end
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // RAM control signals - only read when accessing valid (non-padding) pixels
    assign ram_re = (current_state == FETCH_PIXELS) && is_valid_ram_access;
    assign ram_addr = word_addr[$clog2(MAX_IMG_W*MAX_IMG_H*MAX_CHANNELS/16)-1:0];
    
    // Output to sliding window
    assign patch_A0_out = assembled_patch[0];
    assign patch_A1_out = assembled_patch[1];
    assign patch_A2_out = assembled_patch[2];
    assign patch_A3_out = assembled_patch[3];
    assign patch_valid = (current_state == PATCH_AVAILABLE);
    assign patch_ready = (current_state == PATCH_AVAILABLE);
    
    // Extraction complete signal - when we've processed all output positions (input_size x input_size)
    assign extraction_complete = (current_state == ADVANCE_POSITION) && 
                                ((patch_col + STRIDE) >= img_width) &&
                                ((patch_row + STRIDE) >= img_height);

endmodule 
