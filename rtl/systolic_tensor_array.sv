`include "sys_types.svh"

module systolic_tensor_array (
    input   logic    clk             // System clock, rising-edge
    ,input  logic    reset           // Active-high synchronous reset
    ,input  logic    stall           // Freezes array at current computation when not enough input

    // A matrix inputs: one 4-wide int8 vector per row, left inputs
    ,input  int8_t   A0 [0:3]        // Row 0 of A
    ,input  int8_t   A1 [0:3]        // Row 1 of A
    ,input  int8_t   A2 [0:3]        // Row 2 of A
    ,input  int8_t   A3 [0:3]        // Row 3 of A

    // B matrix inputs: one 4-wide int8 vector per column, top inputs
    ,input  int8_t   B0 [0:3]        // Column 0 of B
    ,input  int8_t   B1 [0:3]        // Column 1 of B
    ,input  int8_t   B2 [0:3]        // Column 2 of B
    ,input  int8_t   B3 [0:3]        // Column 3 of B

    // Per-PE load_bias: when high, accumulator set to corresponding bias value
    ,input  logic   load_bias0 [0:3] // Row 0 load_bias for cols 0..3
    ,input  logic   load_bias1 [0:3] // Row 1 load_bias
    ,input  logic   load_bias2 [0:3] // Row 2 load_bias
    ,input  logic   load_bias3 [0:3] // Row 3 load_bias

    // Per-PE bias inputs: int32 biases preloaded into accumulator
    ,input  int32_t bias0 [0:3]      // Bias for row 0 PEs
    ,input  int32_t bias1 [0:3]      // Bias for row 1 PEs
    ,input  int32_t bias2 [0:3]      // Bias for row 2 PEs
    ,input  int32_t bias3 [0:3]      // Bias for row 3 PEs

    // PE mask to indicate which PEs are active for current tile
    ,input  logic   pe_mask [0:N*N-1] // 1 if PE should be active, 0 if ignored

    // Final outputs: one 4-wide int32 vector per row of PEs, accumulator value for each PE
    ,output int32_t C0 [0:3]         // Outputs from row 0 PEs
    ,output int32_t C1 [0:3]         // Outputs from row 1 PEs
    ,output int32_t C2 [0:3]         // Outputs from row 2 PEs
    ,output int32_t C3 [0:3]         // Outputs from row 3 PEs

    // Idle signal: high when no non-zero inputs or register values (no activity)
    ,output logic   sta_idle         // High when STA is idle
);
  
    // Systolic array height/width (NxN PEs)
    parameter int N = 4;

    // Tile size for pipelining across boundaries; Tried 1=1x4x1, 2=2x4x2, 4=4x4x4 STA tile
    parameter int TILE_SIZE = 1;

    // How many ops each PE handles at once
    parameter int VECTOR_WIDTH = 4;  

    // Internal data buses
    // Meta comments required to avoid UNOPTFLAT warning
    // Veri[lator] doesn't like cascading assignment of A/B_data used in generate
    /*verilator lint_off UNOPTFLAT*/
    int8_t A_data [0:N-1][0:N-1][0:VECTOR_WIDTH-1];
    int8_t B_data [0:N-1][0:N-1][0:VECTOR_WIDTH-1];

    // PE outputs for connecting to final outputs
    int32_t pe_outputs [N][N];

    logic lb_pe; // Intermediate signal used to check if any PEs are loading bias

    // Feed in row/col data from edges
    assign A_data[0][0] = A0;  assign B_data[0][0] = B0;
    assign A_data[1][0] = A1;  assign B_data[0][1] = B1;
    assign A_data[2][0] = A2;  assign B_data[0][2] = B2;
    assign A_data[3][0] = A3;  assign B_data[0][3] = B3;

    // Connect processing elements and place intermediate registers
    generate
        for (genvar row = 0; row < N; row++) begin : gen_rows
            for (genvar col = 0; col < N; col++) begin : gen_cols
                localparam row_tile_boundary = (row % TILE_SIZE == 0);
                localparam col_tile_boundary = (col % TILE_SIZE == 0);

                // A_data (left -> right)
                if (col != 0) begin : gen_a_data
                    if (col_tile_boundary) begin : gen_col_reg
                        always_ff @(posedge clk) begin
                            if (reset)
                                A_data[row][col] <= '{default: 'b0};
                            else if (stall)
                                A_data[row][col] <= A_data[row][col];  
                            else
                                A_data[row][col] <= A_data[row][col-1];
                        end
                    end else begin : gen_col_wire
                        assign A_data[row][col] = A_data[row][col-1];
                    end
                end

                // B_data (top -> bottom)
                if (row != 0) begin : gen_b_data
                    if (row_tile_boundary) begin : gen_row_reg
                        always_ff @(posedge clk) begin
                            if (reset)
                                B_data[row][col] <= '{default: 'b0};
                            else if (stall)
                                B_data[row][col] <= B_data[row][col];
                            else
                                B_data[row][col] <= B_data[row-1][col];
                        end
                    end else begin : gen_row_wire
                        assign B_data[row][col] = B_data[row-1][col];
                    end
                end

                logic lb;
                int32_t b_in;
                // Select control signals for bias load for this PE
                assign lb = (row==0) ? load_bias0[col] :
                            (row==1) ? load_bias1[col] :
                            (row==2) ? load_bias2[col] :
                                       load_bias3[col];

                assign b_in = (row==0) ? bias0[col] :
                              (row==1) ? bias1[col] :
                              (row==2) ? bias2[col] :
                                         bias3[col];

                // Tensor PE: computes 4-dot-product + accumulation
                tensor_process_elem pe (
                    .clk(clk)
                    ,.reset(reset)
                    ,.load_bias(lb)
                    ,.bias_in(b_in)
                    ,.stall(stall)
                    ,.left_in(A_data[row][col])
                    ,.top_in(B_data[row][col])
                    ,.sum_out(pe_outputs[row][col])
                );
            end
        end
    endgenerate

    // Connect C_out to PE outputs
    assign C0 = pe_outputs[0];
    assign C1 = pe_outputs[1];
    assign C2 = pe_outputs[2];
    assign C3 = pe_outputs[3];

    // Idle detection logic
    always_comb begin
        sta_idle = 1'b1; // Start assuming idle
        
        // Check if any valid PE has non-zero inputs or is loading bias
        for (int row = 0; row < N; row++) begin
            for (int col = 0; col < N; col++) begin
                int flat_idx = row * N + col;
                
                // Only check PEs that are valid for current tile
                if (pe_mask[flat_idx]) begin
                    // Check if any A inputs are non-zero
                    for (int k = 0; k < VECTOR_WIDTH; k++) begin
                        if (A_data[row][col][k] != 8'b0) begin
                            sta_idle = 1'b0;
                        end
                    end
                    
                    // Check if any B inputs are non-zero
                    for (int k = 0; k < VECTOR_WIDTH; k++) begin
                        if (B_data[row][col][k] != 8'b0) begin
                            sta_idle = 1'b0;
                        end
                    end
                    
                    // Check if bias loading is active
                    lb_pe       = (row==0) ? load_bias0[col] :
                                  (row==1) ? load_bias1[col] :
                                  (row==2) ? load_bias2[col] :
                                             load_bias3[col];
                    if (lb_pe) begin
                        sta_idle = 1'b0;
                    end
                end
            end
        end
    end

    /*verilator lint_on UNOPTFLAT*/

endmodule
