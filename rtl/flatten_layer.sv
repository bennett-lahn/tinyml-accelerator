`include "sys_types.svh"

module flatten_layer #(
    parameter INPUT_HEIGHT = 2,
    parameter INPUT_WIDTH = 2, 
    parameter INPUT_CHANNELS = 64,
    parameter CHUNK_SIZE = 16,  // 128 bits = 16 int8_t values
    parameter TOTAL_CHUNKS = (INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS) / CHUNK_SIZE,  // 256/16 = 16 chunks
    parameter OUTPUT_SIZE = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS  // 256
)(
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_flatten,
    input logic output_read_enable,  // Request next pixel
    
    // Input data (128-bit chunk)
    input logic [127:0] input_chunk,  // 128-bit vector = 16 bytes
    input logic chunk_valid,  // Chunk data is valid
    
    // Output data (one pixel at a time)
    output int8_t output_data,
    output logic [$clog2(OUTPUT_SIZE)-1:0] output_addr,  // Current output address
    output logic output_valid,
    output logic flatten_complete,
    
    // Memory request signals
    output logic request_chunk,  // Request next 128-bit chunk
    output logic [$clog2(TOTAL_CHUNKS)-1:0] chunk_addr  // Current chunk address (0-15)
);

    // Current chunk buffer (only holds 16 bytes)
    int8_t chunk_buffer [0:CHUNK_SIZE-1];
    
    // State machine
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        READ_CHUNK = 2'b01,
        OUTPUT_CHUNK = 2'b10,
        COMPLETE = 2'b11
    } flatten_state_t;
    
    flatten_state_t current_state, next_state;
    
    // Counters
    logic [$clog2(TOTAL_CHUNKS)-1:0] current_chunk_addr;
    logic [$clog2(CHUNK_SIZE)-1:0] pixel_index;  // Index within current chunk (0-15)
    logic [$clog2(OUTPUT_SIZE)-1:0] total_pixel_addr;  // Overall pixel address (0-255)
    
    // State machine logic
    always_ff @(posedge clk) begin
        if (reset) begin
            current_state <= IDLE;
            current_chunk_addr <= 0;
            pixel_index <= 0;
            total_pixel_addr <= 0;
            for (int i = 0; i < CHUNK_SIZE; i++) begin
                chunk_buffer[i] <= 0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    if (start_flatten) begin
                        current_chunk_addr <= 0;
                        pixel_index <= 0;
                        total_pixel_addr <= 0;
                        for (int i = 0; i < CHUNK_SIZE; i++) begin
                            chunk_buffer[i] <= 0;
                        end
                    end
                end
                
                READ_CHUNK: begin
                    if (chunk_valid) begin
                        // Unpack 128-bit vector into 16 bytes
                        for (int i = 0; i < CHUNK_SIZE; i++) begin
                            chunk_buffer[i] <= int8_t'(input_chunk[i*8 +: 8]);
                        end
                        pixel_index <= 0;  // Reset pixel index for new chunk
                    end
                end
                
                OUTPUT_CHUNK: begin
                    if (pixel_index < CHUNK_SIZE - 1) begin
                        // Move to next pixel in chunk
                        pixel_index <= pixel_index + 1;
                        total_pixel_addr <= total_pixel_addr + 1;
                    end else if (pixel_index >= CHUNK_SIZE - 1) begin
                        // Finished current chunk, prepare for next
                        pixel_index <= 0;
                        total_pixel_addr <= total_pixel_addr + 1;
                        if (current_chunk_addr < TOTAL_CHUNKS - 1) begin
                            current_chunk_addr <= current_chunk_addr + 1;
                        end
                    end
                end
                
                COMPLETE: begin
                    // Stay here until next start_flatten
                    if (start_flatten) begin
                        current_chunk_addr <= 0;
                        pixel_index <= 0;
                        total_pixel_addr <= 0;
                    end
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start_flatten) begin
                    next_state = READ_CHUNK;
                end
            end
            
            READ_CHUNK: begin
                if (chunk_valid) begin
                    next_state = OUTPUT_CHUNK;
                end
            end
            
            OUTPUT_CHUNK: begin
                // Check if we've output all pixels in current chunk
                if (pixel_index >= CHUNK_SIZE - 1) begin
                    if (current_chunk_addr >= TOTAL_CHUNKS - 1) begin
                        // All chunks processed
                        next_state = COMPLETE;
                    end else begin
                        // More chunks to process
                        next_state = READ_CHUNK;
                    end
                end
            end
            
            COMPLETE: begin
                if (start_flatten) begin
                    next_state = READ_CHUNK;
                end
            end
        endcase
    end
    
    // Output assignments
    assign output_data = chunk_buffer[pixel_index];
    assign output_addr = total_pixel_addr;
    assign output_valid = (current_state == OUTPUT_CHUNK);
    assign flatten_complete = (current_state == COMPLETE);
    assign request_chunk = (current_state == READ_CHUNK);
    assign chunk_addr = current_chunk_addr;

endmodule 
