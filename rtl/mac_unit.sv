`include "sys_types.svh"

module mac_unit (
    input logic clk,
    input logic reset,
    input logic enable,
    input logic load_bias,
    input logic [7:0] input_data,
    input logic [7:0] weight_data,
    input logic [31:0] bias_in,
    input logic [31:0] partial_sum_in,
    output logic [31:0] partial_sum_out,
    output logic output_valid
);

    logic [31:0] accumulator;
    logic [15:0] mult_result;
    logic enable_reg;

    // Multiply input and weight (both signed 8-bit)
    always_comb begin
        mult_result = $signed(input_data) * $signed(weight_data);
    end

    // Register enable signal for output_valid timing
    always_ff @(posedge clk) begin
        if (reset) begin
            enable_reg <= 1'b0;
        end else begin
            enable_reg <= enable;
        end
    end

    // Accumulator logic
    always_ff @(posedge clk) begin
        if (reset) begin
            accumulator <= 32'b0;
        end else if (load_bias) begin
            accumulator <= bias_in;
        end else if (enable) begin
            accumulator <= partial_sum_in + {{16{mult_result[15]}}, mult_result};
        end
    end

    // Output assignments
    assign partial_sum_out = accumulator;
    assign output_valid = enable_reg;

endmodule 
