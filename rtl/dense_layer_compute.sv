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
    
    // Output (before activation)
    output logic [31:0] output_vector [63:0],
    output logic computation_complete,
    output logic output_valid,
    
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
        COMPUTE_MAC = 3'b010,
        COMPLETE = 3'b011
    } dense_state_t;
    
    dense_state_t current_state, next_state;
    
    // Computation counters
    logic [7:0] input_idx;   // Current input index
    logic [5:0] output_idx;  // Current output index
    logic [5:0] bias_idx;    // For loading bias values
    logic computation_done;  // Flag to indicate computation is complete
    logic final_computation; // Flag to indicate we're on the final computation
    logic [1:0] completion_delay; // Delay counter for final computation
    logic final_input_processed; // Flag to indicate final input was processed
    
    // Track which output the MAC result belongs to (to handle timing delays)
    logic [5:0] mac_result_output_idx;
    
    // Delay counter to wait for final MAC result
    logic [2:0] final_mac_delay;
    
    // Pipeline to track which output each MAC result belongs to
    logic [5:0] mac_output_pipeline [1:0];  // 2-stage pipeline for 2-cycle MAC delay
    
    // Accumulation registers for each output neuron
    logic [31:0] output_accumulator [63:0];
    
    // TPE interface signals
    logic mac_enable;
    logic mac_load_bias;
    logic [7:0] mac_input_data;
    logic [7:0] mac_weight_data;
    logic [31:0] mac_bias_in;
    logic [31:0] mac_partial_sum_in;
    logic [31:0] mac_partial_sum_out;
    logic mac_output_valid;
    
    // MAC control signals
    logic mac_reset;
    
    // Instantiate MAC Unit
    mac_unit mac_inst (
        .clk(clk),
        .reset(mac_reset),
        .enable(mac_enable),
        .load_bias(mac_load_bias),
        .input_data(mac_input_data),
        .weight_data(mac_weight_data),
        .bias_in(mac_bias_in),
        .partial_sum_in(mac_partial_sum_in),
        .partial_sum_out(mac_partial_sum_out),
        .output_valid(mac_output_valid)
    );
    
    // Memory address generation
    always_comb begin
        // Tensor RAM address (input vector)
        tensor_ram_addr = input_idx;
        
        // Weight ROM address (row-major: input_idx * output_size + output_idx)
        weight_rom_addr = (input_idx * output_size[5:0]) + {{8{1'b0}}, output_idx};
        
        // Bias ROM address
        bias_rom_addr = bias_idx;
    end
    
    // Memory read enable signals
    always_comb begin
        tensor_ram_re = (current_state == COMPUTE_MAC) && (input_idx < input_size[7:0]) && (output_idx < output_size[5:0]);
        weight_rom_re = (current_state == COMPUTE_MAC) && (input_idx < input_size[7:0]) && (output_idx < output_size[5:0]);
        bias_rom_re = (current_state == LOAD_BIAS) && (bias_idx < output_size[5:0]);
    end
    
    // MAC control signals
    always_comb begin
        mac_enable = (current_state == COMPUTE_MAC) && (input_idx < input_size[7:0]) && (output_idx < output_size[5:0]);
        mac_load_bias = (current_state == COMPUTE_MAC) && (input_idx == 0); // Load bias at start of each output neuron
        mac_input_data = tensor_ram_dout;
        mac_weight_data = weight_rom_dout;
        mac_bias_in = output_accumulator[output_idx]; // Bias value loaded during LOAD_BIAS state
        mac_partial_sum_in = mac_partial_sum_out; // Use MAC's own accumulator for chaining
        
        // Reset MAC unit when transitioning between outputs or at global reset
        mac_reset = reset || (final_input_processed && (output_idx < output_size[5:0] - 1));
    end
    
    // Main state machine
    always_ff @(posedge clk) begin
        if (reset) begin
            current_state <= IDLE;
            input_idx <= 0;
            output_idx <= 0;
            bias_idx <= 0;
            computation_done <= 0;
            final_computation <= 0;
            completion_delay <= 0;
            final_input_processed <= 0;
            final_mac_delay <= 0;
            
            // Initialize pipeline
            mac_output_pipeline[0] <= 0;
            mac_output_pipeline[1] <= 0;
            
            // Initialize accumulators
            for (int i = 0; i < 64; i++) begin
                output_accumulator[i] <= 0;
            end
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    if (start_compute && input_valid) begin
                        input_idx <= 0;
                        output_idx <= 0;
                        bias_idx <= 0;
                        computation_done <= 0;
                        final_computation <= 0;
                        completion_delay <= 0;
                        final_input_processed <= 0;
                        final_mac_delay <= 0;
                        
                        // Initialize pipeline
                        mac_output_pipeline[0] <= 0;
                        mac_output_pipeline[1] <= 0;
                        
                        // Clear accumulators
                        for (int i = 0; i < 64; i++) begin
                            output_accumulator[i] <= 0;
                        end
                    end
                end
                
                LOAD_BIAS: begin
                    // Load bias values into accumulators (with one cycle delay for memory read)
                    if (bias_idx > 0 && bias_idx <= output_size[5:0]) begin
                        output_accumulator[bias_idx - 1] <= bias_rom_dout;
                    end
                    
                    // Increment bias index
                    if (bias_idx <= output_size[5:0]) begin
                        bias_idx <= bias_idx + 1;
                    end
                    
                    // Reset computation indices when bias loading is done
                    if (bias_idx > output_size[5:0]) begin
                        input_idx <= 0;
                        output_idx <= 0;
                    end
                end
                
                COMPUTE_MAC: begin
                    if (mac_output_valid) begin
                        // Store MAC results as they become available
                        output_accumulator[output_idx] <= mac_partial_sum_out;
                        
                        // Check if this is the final computation for current output
                        if (input_idx == input_size[7:0] - 1) begin
                            // This is the final input for current output
                            final_input_processed <= 1;
                        end else begin
                            // Move to next input
                            input_idx <= input_idx + 1;
                        end
                    end
                    
                    // Handle completion after final result is stored
                    if (final_input_processed) begin
                        if (output_idx == output_size[5:0] - 1) begin
                            // Finished all outputs - set completion flag
                            computation_done <= 1;
                        end else begin
                            // Move to next output (MAC will be reset automatically)
                            input_idx <= 0;
                            output_idx <= output_idx + 1;
                        end
                        final_input_processed <= 0;
                    end
                end
                
                COMPLETE: begin
                    // Computation complete, outputs ready
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
                if (start_compute && input_valid) begin
                    next_state = LOAD_BIAS;
                end
            end
            
            LOAD_BIAS: begin
                if (bias_idx > output_size[5:0]) begin
                    next_state = COMPUTE_MAC;
                end
            end
            
            COMPUTE_MAC: begin
                // Check if all computations are done using the completion flag
                if (computation_done) begin
                    next_state = COMPLETE;
                end
            end
            
            COMPLETE: begin
                if (start_compute) begin
                    next_state = LOAD_BIAS;
                end else begin
                    next_state = IDLE;
                end
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output assignments
    assign output_vector = output_accumulator;
    assign computation_complete = (current_state == COMPLETE);
    assign output_valid = computation_complete;

endmodule 
