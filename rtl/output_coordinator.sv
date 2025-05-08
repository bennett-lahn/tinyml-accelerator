module output_coordinator #(
  parameter int ROWS    = 4,                  // # of PE rows
  parameter int COLS    = 4,                  // # of PE columns
  parameter int MAX_N   = 16,                 // max matrix dimension
  parameter int N_BITS  = $clog2(MAX_N+1),    // bits to hold mat_size & coords
  // worst-case delay = (ROWS-1)+(COLS-1)+ceil(MAX_N/4)
  parameter int CT_BITS = $clog2((ROWS-1)+(COLS-1)+((MAX_N+3)/4)+1)
)(
  input  logic               clk
  ,input  logic              reset
  ,input  logic [N_BITS-1:0] mat_size       // N = 3…MAX_N; Used to calculate number of compute cycles
  ,input  logic              input_valid    // Inject new value at PE[0,0] of
  ,input  logic [N_BITS-1:0] pos_row        // block’s base row
  ,input  logic [N_BITS-1:0] pos_col        // block’s base col

  ,output logic              out_valid [0:ROWS-1][0:COLS-1]
  ,output logic [N_BITS-1:0] out_row   [0:ROWS-1][0:COLS-1]
  ,output logic [N_BITS-1:0] out_col   [0:ROWS-1][0:COLS-1] // TODO: Rewrite module outputs to support cocotb (no packed multi-dimensional arrays)
);

  // Number of cycles per PE to finish MACs, equivalent to ceil(mat_size/4)
  wire [CT_BITS-1:0] compute_cycles = (mat_size + 3) >> 2;

  // per-PE state
  logic [CT_BITS-1:0] initial_cnt [0:ROWS-1][0:COLS-1];
  logic [CT_BITS-1:0] curr_cnt    [0:ROWS-1][0:COLS-1];
  logic               active      [0:ROWS-1][0:COLS-1];
  logic [N_BITS-1:0]  base_row    [0:ROWS-1][0:COLS-1];
  logic [N_BITS-1:0]  base_col    [0:ROWS-1][0:COLS-1];

  always_ff @(posedge clk) begin
    if (reset) begin
      // Clear all PEs
      for (int i = 0; i < ROWS; i++) begin
        for (int j = 0; j < COLS; j++) begin
          initial_cnt[i][j] <= '0;
          curr_cnt   [i][j] <= '0;
          active     [i][j] <= 1'b0;
          base_row   [i][j] <= '0;
          base_col   [i][j] <= '0;
        end
      end
    end else begin

      // Countdown and retire each PE
      for (int i = 0; i < ROWS; i++) begin
        for (int j = 0; j < COLS; j++) begin
          if (active[i][j]) begin
            if (curr_cnt[i][j] != 0) begin
              curr_cnt[i][j] <= curr_cnt[i][j] - 1;
            end else begin
              // finished this cycle → clear next
              active[i][j] <= 1'b0;
            end
          end
        end
      end

      // Inject new block only into PE(0,0)
      if (input_valid) begin
        initial_cnt [0][0] <= compute_cycles;
        curr_cnt    [0][0] <= compute_cycles - 1;
        active      [0][0] <= 1'b1;
        base_row    [0][0] <= pos_row;
        base_col    [0][0] <= pos_col;
      end

      // Propagate down column 0 from above neighbor
      for (int i = 1; i < ROWS; i++) begin
        // when PE(i-1,0) has just started (curr==initial)
        if (active[i-1][0] && (curr_cnt[i-1][0] == initial_cnt[i-1][0] - 1)) begin
          initial_cnt [i][0] <= initial_cnt[i-1][0];
          curr_cnt    [i][0] <= initial_cnt[i-1][0] - 1;
          active      [i][0] <= 1'b1;
          base_row    [i][0] <= base_row[i-1][0];
          base_col    [i][0] <= base_col[i-1][0];
        end
      end

      // Propagate right in each row from left neighbor
      for (int i = 0; i < ROWS; i++) begin
        for (int j = 1; j < COLS; j++) begin
          // when PE(i,j-1) has just started
          if (active[i][j-1] && (curr_cnt[i][j-1] == initial_cnt[i][j-1] - 1)) begin;
            initial_cnt [i][j] <= initial_cnt[i][j-1];
            curr_cnt    [i][j] <= initial_cnt[i][j-1] - 1;
            active      [i][j] <= 1'b1;
            base_row    [i][j] <= base_row[i][j-1];
            base_col    [i][j] <= base_col[i][j-1];
          end
        end
      end
    end
  end

  // Output valid & coordinates when outputs are valid
  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      for (int j = 0; j < COLS; j++) begin
        out_valid[i][j] = active[i][j] && (curr_cnt[i][j] == 0);
        out_row   [i][j] = base_row[i][j] + i;
        out_col   [i][j] = base_col[i][j] + j;
      end
    end
  end

endmodule

// Memory:
// Store rows of A (input) for easy reading
// Store columns of B (weights) for easy reading

// Tiling:
// I think it is feasible (if not the best idea) to stream all values in for an array larger than 16x16 and calculate the final output in place in PEs
// AI says this is not feasible and for a larger array (say 512x512) we must tile both MxN and reductions (output dimension) into multiple calcs
  // AKA every array needs to be split into smaller 16x16 arrays
