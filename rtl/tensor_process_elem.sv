`include "sys_types.svh"

// ======================================================================================================
// TENSOR PROCESS ELEMENT
// ======================================================================================================
// This module implements a single processing element (PE) for the systolic tensor array in the
// TinyML accelerator. It performs vectorized multiply-accumulate (MAC) operations with bias
// loading capabilities, serving as the fundamental computation unit in the 4x4 systolic array.
//
// FUNCTIONALITY:
// - Performs 4-element vectorized dot product per cycle
// - Accumulates results over multiple computation cycles
// - Supports bias loading to initialize the accumulator
// - Outputs 32-bit accumulated result for downstream processing
//
// COMPUTATION:
// - Multiplies corresponding elements from left_in and top_in arrays
// - Sums all 4 multiplication results with current accumulator value
// - Performs: sum = accumulator + (left_in[0]*top_in[0] + ... + left_in[3]*top_in[3])
// - Updates accumulator register with new sum on each clock cycle
//
// CONTROL LOGIC:
// - Reset: Clears accumulator to zero
// - Load_bias: Loads bias_in value into accumulator (for layer initialization)
// - Stall: Freezes accumulator at current value (for flow control)
// - Normal operation: Accumulates MAC results
// - Priority: reset > load_bias > stall > MAC calculation
//
// DATA INTERFACE:
// - left_in[0:3]: 4 int8 values from left neighbor (matrix A elements)
// - top_in[0:3]: 4 int8 values from top neighbor (matrix B elements)
// - bias_in: 32-bit bias value for layer initialization
// - sum_out: 32-bit accumulated result for output
//
// ARCHITECTURE:
// - 4 parallel 8x8 multipliers for vectorized computation
// - 32-bit accumulator register for result storage
// - Combinational adder tree for sum calculation
// - Synchronous control logic with priority-based operation
//
// INTEGRATION:
// - Used by systolic_tensor_array as individual processing elements
// - Arranged in 4x4 grid for matrix multiplication operations
// - Receives systolic data flow from neighboring PEs
// - Outputs to array-level result collection
//
// TIMING:
// - Single-cycle MAC operation with accumulation
// - Synchronous reset and control signal handling
// - Stall capability for pipeline flow control
// - Bias loading for layer initialization
// ======================================================================================================

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
