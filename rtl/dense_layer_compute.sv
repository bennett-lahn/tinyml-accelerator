`include "sys_types.svh"

module dense_layer_compute (
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_compute,
    input logic input_valid,
    
    // Runtime configuration inputs
    input logic [$clog2(256+1)-1:0] input_size,    // Actual input size (up to 256)
    input logic [$clog2(64+1)-1:0] output_size,    // Actual output size (up to 64)
    
    // Tensor RAM interface (for input vectors)
    output logic [$clog2(256)-1:0] tensor_ram_addr,
    output logic tensor_ram_re,
    input logic [7:0] tensor_ram_dout,
    
    // Weight ROM interface (for weight matrix)
    output logic [$clog2(256*64)-1:0] weight_rom_addr,
    output logic weight_rom_re,
    input logic [7:0] weight_rom_dout,
    
    // Bias ROM interface (for bias vectors)
    output logic [$clog2(64)-1:0] bias_rom_addr,
    output logic bias_rom_re,
    input logic [31:0] bias_rom_dout,
    
    // Single output (one result at a time)
    output logic [31:0] output_data,
    output logic [$clog2(64)-1:0] output_channel,  // Which output channel this result is for
    output logic [$clog2(64)-1:0] output_addr,     // Address of calculated value (counts up from 0)
    output logic output_ready,                     // Indicates output_data is valid for current channel
    output logic computation_complete,             // All outputs computed
    
    // Status outputs
    output logic [$clog2(256+1)-1:0] current_input_size,
    output logic [$clog2(64+1)-1:0] current_output_size
);

    // Echo back the input configuration
    assign current_input_size = input_size;
    assign current_output_size = output_size;

    // State machine for dense computation
    typedef enum logic [2:0] {
        IDLE = 3'b000,
        LOAD_BIAS = 3'b001,
        LOAD_DATA = 3'b010,
        COMPUTE_MAC = 3'b011,
        OUTPUT_READY = 3'b100,
        COMPLETE = 3'b101
    } dense_state_t;
    
    dense_state_t current_state, next_state;
    
    // Computation counters
    logic [$clog2(256)-1:0] input_idx;   // Current input index within current output
    logic [$clog2(64)-1:0] output_idx;   // Current output index
    logic [$clog2(64)-1:0] addr_counter;  // Simple address counter (0, 1, 2, ...)
    logic computation_done;              // Flag to indicate computation is complete
    
    // MAC interface signals
    logic mac_load_bias;
    logic mac_enable;
    int32_t mac_bias_in;
    int8_t mac_left_in;
    int8_t mac_top_in;
    int32_t mac_sum_out;
    
    // MAC control signals
    logic mac_reset;
    
    // Instantiate MAC Unit
    mac_unit mac_inst (
        .clk(clk),
        .reset(mac_reset),
        .enable(mac_enable),
        .load_bias(mac_load_bias),
        .bias_in(mac_bias_in),
        .left_in(mac_left_in),
        .top_in(mac_top_in),
        .sum_out(mac_sum_out)
    );
    
    // Memory address generation
    always_comb begin
        // Tensor RAM address (input vector)
        tensor_ram_addr = input_idx;
        
        // Weight ROM address (row-major addressing: output_idx * input_size + input_idx)
        weight_rom_addr = output_idx * input_size[$clog2(256)-1:0] + input_idx;
        
        // Bias ROM address
        bias_rom_addr = output_idx;
    end
    
    // Memory read enable signals
    always_comb begin
        // Start tensor/weight reads in LOAD_DATA to ensure data is valid in COMPUTE_MAC
        tensor_ram_re = (current_state == LOAD_DATA);
        weight_rom_re = (current_state == LOAD_DATA);
        // Read bias ROM during LOAD_BIAS state (output_idx is now properly updated)
        bias_rom_re = (current_state == LOAD_BIAS);
    end
    
    // MAC input assignments
    always_comb begin
        mac_left_in = int8_t'(tensor_ram_dout);
        mac_top_in = int8_t'(weight_rom_dout);
        mac_bias_in = int32_t'(bias_rom_dout);
        // mac_load_bias assigned in next state logic for proper timing
        // Reset MAC in multiple scenarios for robust operation:
        // 1. System reset
        // 2. When idle (no computation active)  
        // 3. When transitioning from OUTPUT_READY to LOAD_BIAS (between outputs)
        // 4. When starting new computation from IDLE
        mac_reset = reset || 
                   (current_state == IDLE) || 
                   ((current_state == OUTPUT_READY) && (next_state == LOAD_BIAS)) ||
                   ((current_state == IDLE) && (next_state == LOAD_BIAS));
    end
    
    // Main state machine
    always_ff @(posedge clk) begin
        if (reset) begin
            current_state <= IDLE;
            input_idx <= 0;
            output_idx <= 0;
            addr_counter <= 0;
            computation_done <= 0;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    if (start_compute && input_valid) begin
                        input_idx <= 0;
                        output_idx <= 0;
                        addr_counter <= 0;
                        computation_done <= 0;
                    end
                end
                
                LOAD_BIAS: begin
                    // Bias is loaded into MAC on this cycle
                    // No additional logic needed here as MAC handles the loading
                    // Ensure clean state for output computation
                    if (input_idx != 0) begin
                        input_idx <= 0;
                    end
                end
                
                LOAD_DATA: begin
                    // Transition to COMPUTE_MAC
                    // Memory reads have been initiated, data will be valid next cycle
                end
                
                COMPUTE_MAC: begin
                    // MAC operation happens with stable addresses
                    // Index incrementing moved to next state logic for proper timing
                    
                    // Check if this was the last input for current output
                    if (input_idx == input_size[$clog2(256)-1:0] - 1) begin
                        // Finished processing all inputs for current output
                        // Reset input index for next output and handle output completion
                        input_idx <= 0;
                        
                        if (output_idx == output_size[$clog2(64)-1:0] - 1) begin
                            // All outputs computed
                            computation_done <= 1;
                        end
                    end else begin
                        // Increment input index when moving to next input
                        input_idx <= input_idx + 1;
                    end
                end
                
                OUTPUT_READY: begin
                    // Output is ready for downstream module to consume
                    // Stay in this state for one cycle to allow downstream to capture
                    // Increment output_idx and addr_counter here AFTER output is captured
                    if (output_idx < output_size[$clog2(64)-1:0] - 1) begin
                        output_idx <= output_idx + 1;
                        addr_counter <= addr_counter + 1;
                        // Reset input_idx for next output computation
                        input_idx <= 0;
                    end
                end
                
                COMPLETE: begin
                    // Computation complete
                    if (!start_compute) begin
                        computation_done <= 0;
                    end
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
        mac_load_bias = 0; // Default value
        mac_enable = 0;    // Default value - only enable during COMPUTE_MAC
        
        case (current_state)
            IDLE: begin
                if (start_compute && input_valid) begin
                    next_state = LOAD_BIAS;
                end
            end
            
            LOAD_BIAS: begin
                // Transition to LOAD_DATA (bias ROM read happens during this state)
                next_state = LOAD_DATA;
            end
            
            LOAD_DATA: begin
                // Transition to COMPUTE_MAC
                next_state = COMPUTE_MAC;
                // Load bias into MAC when transitioning to COMPUTE_MAC for first input only
                if (input_idx == 0) begin
                    mac_load_bias = 1;
                end
            end
            
            COMPUTE_MAC: begin
                // Enable MAC accumulation only during this state
                mac_enable = 1;
                
                if (input_idx == input_size[$clog2(256)-1:0] - 1) begin
                    // Finished all inputs for current output
                    next_state = OUTPUT_READY;
                end else begin
                    // More inputs to process, go back to LOAD_DATA for next input
                    next_state = LOAD_DATA;
                end
            end
            
            OUTPUT_READY: begin
                if (output_idx == output_size[$clog2(64)-1:0] - 1) begin
                    // This was the last output
                    next_state = COMPLETE;
                end else begin
                    // More outputs to process, load bias for next output
                    next_state = LOAD_BIAS;
                end
            end
            
            COMPLETE: begin
                if (!start_compute) begin
                    next_state = IDLE;
                end else begin
                    // Start new computation
                    next_state = LOAD_BIAS;
                end
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output assignments
    assign output_data = mac_sum_out;
    assign output_channel = output_idx;
    assign output_addr = addr_counter;
    assign output_ready = (current_state == OUTPUT_READY);
    assign computation_complete = (current_state == COMPLETE);

endmodule 
