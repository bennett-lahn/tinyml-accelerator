`include "tensor_process_elem.sv"

module systolic_tensor_array #(
    parameter int N = 8,               // Systolic array height/width (NxN PEs)
    parameter int TILE_SIZE = 2,       // Tile size for pipelining across boundaries
    parameter int VECTOR_WIDTH = 4     // How many ops each PE handles at once
)(
    input  logic clk,
    input  logic reset,

    input  int8_t A_in [N-1:0][VECTOR_WIDTH-1:0], // One vector per row input
    input  int8_t B_in [N-1:0][VECTOR_WIDTH-1:0], // One vector per column input

    input  logic load_sum [N-1:0][N-1:0],         // Per-PE load signal
    output int32_t C_out [N-1:0][N-1:0]           // Final results
);

    // Internal data buses
    int8_t  A_data_wire [0:N][N-1:0][VECTOR_WIDTH-1:0];
    int8_t  A_data_reg  [0:N][N-1:0][VECTOR_WIDTH-1:0];
    int8_t  A_data      [0:N][N-1:0][VECTOR_WIDTH-1:0];

    int8_t  B_data_wire [N-1:0][0:N][VECTOR_WIDTH-1:0];
    int8_t  B_data_reg  [N-1:0][0:N][VECTOR_WIDTH-1:0];
    int8_t  B_data      [N-1:0][0:N][VECTOR_WIDTH-1:0];

    int32_t psum   [0:N][0:N];

    // Feed in row/col data from edges
    generate
        for (genvar i = 0; i < N; i++) begin : feed_inputs
            assign A_data[0][i] = A_in[i];
            assign B_data[i][0] = B_in[i];
        end
    endgenerate

    assign psum[0][0] = 0;

    generate
        for (genvar row = 0; row < N; row++) begin: gen_rows
            for (genvar col = 0; col < N; col++) begin: gen_cols

                localparam row_tile_boundary = (row % TILE_SIZE == 0) && (row > 0);
                localparam col_tile_boundary = (col % TILE_SIZE == 0) && (col > 0);

                // Row direction: A_data
                if (row_tile_boundary) begin : reg_a
                    always_ff @(posedge clk)
                        A_data_reg[row][col] <= A_data[row-1][col];
                    assign A_data[row][col] = A_data_reg[row][col];
                end else begin : pass_a
                    assign A_data[row][col] = A_data[row-1][col];
                end

                // Column direction: B_data
                if (col_tile_boundary) begin : reg_b
                    always_ff @(posedge clk)
                        B_data_reg[row][col] <= B_data[row][col-1];
                    assign B_data[row][col] = B_data_reg[row][col];
                end else begin : pass_b
                    assign B_data[row][col] = B_data[row][col-1];
                end

                // Tensor PE: computes 4-dot-product + accumulation
                tensor_process_elem pe (
                    .clk(clk),
                    .reset(reset),
                    .load_sum(load_sum[row][col]),
                    .sum_in(psum[row][col]),

                    .left_in(A_data[row][col]),
                    .top_in(B_data[row][col]),

                    .right_out(A_data[row+1][col]),
                    .bottom_out(B_data[row][col+1]),
                    .sum_out(psum[row+1][col+1])
                );

                assign C_out[row][col] = psum[row+1][col+1];
            end
        end
    endgenerate

    // 🧠 [Future Extension] Fusion control logic
    // - Use 'mode' signal (Conv2D / DepthwiseConv2D / Bypass)
    // - Use 'channel_group_id' for grouping parallel DW channels
    // - Route/replicate A_in/B_in accordingly

endmodule
