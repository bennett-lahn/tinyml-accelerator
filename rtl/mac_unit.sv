`include "sys_types.svh"
module mac_unit (
 	input  logic clk				  // System clock
    ,input  logic reset				  // Reset high signal
    ,input  logic enable              // When high, enables MAC accumulation
    ,input  logic load_bias           // When high, loads bias value into accumulator
    ,input  int32_t bias_in           // Bias value to load
    ,input  int8_t left_in 	  // 4 inputs from left, least significant is top input, matrix A
    ,input  int8_t top_in       // 4 inputs from top, least significant is left-most input, matrix B
    ,output int32_t sum_out		      // Output from accumulate register
);

	int16_t mult; // Output of multiplier units
	int32_t sum;							// Sum of multiplier units and  accumulator value
	int32_t accumulator;					// Accumulator register

    assign mult = left_in * top_in;
    assign sum = accumulator + int32_t'(mult);

    // Sequential logic for registering inputs
    // Precedence for accumulator value is reset -> load_bias -> MAC calculation
    always_ff @(posedge clk) begin
    	if (reset) begin
    		accumulator <= 'b0;
    	end else begin
            if (load_bias) begin
                accumulator <= bias_in;
            end else if (enable) begin
                accumulator <= sum;
            end
            // If neither load_bias nor enable, accumulator holds its value
        end
    end

    // Output assignment
    assign sum_out = accumulator;

endmodule 
