`include "sys_types.svh"

module flatten_layer #(
    parameter INPUT_HEIGHT = 2,
    parameter INPUT_WIDTH = 2, 
    parameter INPUT_CHANNELS = 64,
    parameter OUTPUT_SIZE = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS  // 256
)(
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_flatten,
    input logic input_valid,
    
    // Input data (streaming from previous layer)
    input int8_t input_data [0:INPUT_CHANNELS-1],  // One spatial position at a time
    input logic [$clog2(INPUT_HEIGHT)-1:0] input_row,
    input logic [$clog2(INPUT_WIDTH)-1:0] input_col,
    
    // Output data (flattened vector)
    output int8_t flattened_data [0:OUTPUT_SIZE-1],
    output logic flatten_complete,
    output logic output_valid
);

    // Internal storage for the flattened vector
    int8_t internal_buffer [0:OUTPUT_SIZE-1];
    
    // State machine
    typedef enum logic [1:0] {
        IDLE,
        COLLECTING,
        COMPLETE
    } flatten_state_t;
    
    flatten_state_t current_state, next_state;
    
    // Counters
    logic [$clog2(INPUT_HEIGHT*INPUT_WIDTH+1)-1:0] spatial_count;
    logic [$clog2(INPUT_HEIGHT*INPUT_WIDTH+1)-1:0] expected_spatial_positions;
    
    assign expected_spatial_positions = INPUT_HEIGHT * INPUT_WIDTH;
    
    // Calculate flattened index for current input
    logic [$clog2(OUTPUT_SIZE)-1:0] base_index;
    assign base_index = (input_row * INPUT_WIDTH + input_col) * INPUT_CHANNELS;
    
    // State machine logic
    always_ff @(posedge clk) begin
        if (reset) begin
            current_state <= IDLE;
            spatial_count <= 0;
            for (int i = 0; i < OUTPUT_SIZE; i++) begin
                internal_buffer[i] <= 0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    if (start_flatten) begin
                        spatial_count <= 0;
                        for (int i = 0; i < OUTPUT_SIZE; i++) begin
                            internal_buffer[i] <= 0;
                        end
                    end
                end
                
                COLLECTING: begin
                    if (input_valid) begin
                        // Store input channels at the correct flattened positions
                        for (int c = 0; c < INPUT_CHANNELS; c++) begin
                            internal_buffer[base_index + c] <= input_data[c];
                        end
                        spatial_count <= spatial_count + 1;
                    end
                end
                
                COMPLETE: begin
                    // Stay in complete state until next start_flatten
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
                    next_state = COLLECTING;
                end
            end
            
            COLLECTING: begin
                if (spatial_count >= expected_spatial_positions) begin
                    next_state = COMPLETE;
                end
            end
            
            COMPLETE: begin
                if (start_flatten) begin
                    next_state = COLLECTING;
                end
            end
        endcase
    end
    
    // Output assignments
    assign flattened_data = internal_buffer;
    assign flatten_complete = (current_state == COMPLETE);
    assign output_valid = flatten_complete;

endmodule 