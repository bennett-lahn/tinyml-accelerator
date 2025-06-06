`include "sys_types.svh"

module tensor_process_elem (
 	input  logic clk				  // System clock
    ,input  logic reset				  // Reset high signal
    ,input  logic load_bias           // When high, loads bias value into accumulator
    ,input  int32_t bias_in           // Bias value to load
    ,input  logic stall               // Freezes computation to current value
    ,input  int8_t left_in [0:3] 	  // 4 inputs from left, least significant is top input, matrix A
    ,input  int8_t top_in [0:3]       // 4 inputs from top, least significant is left-most input, matrix B
    ,output int32_t sum_out		      // Output from accumulate register
);

	int16_t mult0, mult1, mult2, mult3;		// Output of multiplier units
	int32_t sum;							// Sum of multiplier units and  accumulator value
	int32_t accumulator;					// Accumulator register

    assign mult0 = left_in[0] * top_in[0];
    assign mult1 = left_in[1] * top_in[1];
    assign mult2 = left_in[2] * top_in[2];
    assign mult3 = left_in[3] * top_in[3];
    assign sum = accumulator + int32_t'(mult0) + int32_t'(mult1) + int32_t'(mult2) + int32_t'(mult3);

    // Sequential logic for registering inputs
    // Precedence for accumulator value is reset -> load_bias -> stall -> MAC calculation
    always_ff @(posedge clk) begin
    	if (reset) begin
    		accumulator <= 'b0;
    	end else begin
            accumulator <= (load_bias) ? bias_in : (stall) ? accumulator : sum;
        end
    end

    // Output assignment
    assign sum_out = accumulator;

endmodule 
