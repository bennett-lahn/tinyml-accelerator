module systolic_tensor_array #(
    parameter int N = 4,        // Array size (NxN)
    parameter int TILE_SIZE = 2 // Tile size (must divide N)
)(
    input  logic clk,
    input  logic reset,

    input  int8_t A_in [N-1:0][N-1:0],
    input  int8_t B_in [N-1:0][N-1:0],

    input  logic load_sum [N-1:0][N-1:0],

    output int32_t C_out [N-1:0][N-1:0]
);

    // Matrix data wires with pipelining points
    int8_t  A_data [0:N][N-1:0][3:0];
    int8_t  B_data [N-1:0][0:N][3:0];
    int32_t psum   [0:N][0:N];

    // Feed input into left and top
    generate
        for (genvar i = 0; i < N; i++) begin
            assign A_data[0][i] = A_in[i];
            assign B_data[i][0] = B_in[i];
        end
    endgenerate

    assign psum[0][0] = 0;

    // Instantiate PE grid with tiling
    generate
        for (genvar row = 0; row < N; row++) begin
            for (genvar col = 0; col < N; col++) begin

                // Insert pipeline regs between tiles (not within tiles)
                localparam row_tile_boundary = (row % TILE_SIZE == 0) && (row > 0);
                localparam col_tile_boundary = (col % TILE_SIZE == 0) && (col > 0);

                // Buffer registers for A (left-right) — row direction
                if (row_tile_boundary) begin
                    always_ff @(posedge clk) A_data[row][col] <= A_data[row-1][col];
                end else begin
                    assign A_data[row][col] = A_data[row-1][col];
                end

                // Buffer registers for B (top-bottom) — column direction
                if (col_tile_boundary) begin
                    always_ff @(posedge clk) B_data[row][col] <= B_data[row][col-1];
                end else begin
                    assign B_data[row][col] = B_data[row][col-1];
                end

                // Tensor PE instance
                tensor_process_element pe_inst (
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
endmodule

// Testbench for systolic_tensor_array_pipelined
module systolic_tensor_array_tb();

  parameter N = 2;

  logic clk = 0;
  logic reset;

  int8_t A_in [N-1:0][N-1:0];
  int8_t B_in [N-1:0][N-1:0];
  logic  load_sum [N-1:0][N-1:0];
  int32_t C_out [N-1:0][N-1:0];

  systolic_tensor_array_pipelined #(.N(N), .TILE_SIZE(2)) dut (
    .clk(clk),
    .reset(reset),
    .A_in(A_in),
    .B_in(B_in),
    .load_sum(load_sum),
    .C_out(C_out)
  );

  always #5 clk = ~clk;

  initial begin
    reset = 1;
    A_in[0] = '{1, 2};
    A_in[1] = '{3, 4};
    B_in[0] = '{5, 6};
    B_in[1] = '{7, 8};

    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        load_sum[i][j] = 0;

    #10;
    reset = 0;

    // Wait for computation to propagate
    #50;

    // Expected C_out = A x B:
    // [1 2]   [5 6]   => [1*5 + 2*7, 1*6 + 2*8] => [19, 22]
    // [3 4] x [7 8]   => [3*5 + 4*7, 3*6 + 4*8] => [43, 50]

    $display("C_out[0][0] = %0d (Expected: 19)", C_out[0][0]);
    $display("C_out[0][1] = %0d (Expected: 22)", C_out[0][1]);
    $display("C_out[1][0] = %0d (Expected: 43)", C_out[1][0]);
    $display("C_out[1][1] = %0d (Expected: 50)", C_out[1][1]);
    $stop;
  end
endmodule
