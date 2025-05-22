`include "sys_types.svh"

// TODO: Update test_output_coordinator to test channel passthrough

module output_coordinator #(
  parameter int ROWS    = 4                 // # of PE rows
  ,parameter int COLS    = 4                // # of PE columns
  ,parameter int NUM_CH  = 64               // Max channels in any layer
  ,parameter int CH_BITS = $clog2(NUM_CH+1) // Bits to hold channel number
  ,parameter int MAX_N   = 512              // Max matrix dimension calculated
  ,parameter int N_BITS  = $clog2(MAX_N+1)  // Bits to hold mat_size & coords
  // worst-case delay = (ROWS-1)+(COLS-1)+ceil(MAX_N/4)
  ,parameter int CT_BITS = $clog2((ROWS-1)+(COLS-1)+((MAX_N+3)/4)+1)
)(
  input  logic                clk
  ,input  logic               reset
  ,input  logic [N_BITS-1:0]  mat_size       // N = 3…MAX_N; Used to calculate number of compute cycles
  ,input  logic               input_valid    // Inject new value at PE[0,0]
  ,input  logic               stall
  ,input  logic [N_BITS-1:0]  pos_row        // Block’s base row
  ,input  logic [N_BITS-1:0]  pos_col        // Block’s base col
  ,input  logic [CH_BITS-1:0] channel        // Block's channel

  ,output logic               out_valid    [ROWS*COLS] // High if corresponding STA output is valid, complete
  ,output logic [N_BITS-1:0]  out_row      [ROWS*COLS] // Row in channel for each PE
  ,output logic [N_BITS-1:0]  out_col      [ROWS*COLS] // Col in channel for each PE
  ,output logic [CH_BITS-1:0] out_channels [ROWS*COLS] // Current channel in layer for each PE
);

  // Inject new block only into PE(0,0)
  // PE(0,0) corresponds to flat index 0
  localparam int PE00_FLAT_IDX = 0 * COLS + 0;
  localparam int TOTAL_PES = ROWS * COLS;

  // Number of cycles per PE to finish MACs, equivalent to ceil(mat_size/4)
  logic [CT_BITS-1:0] compute_cycles;
  assign compute_cycles = CT_BITS'((mat_size + N_BITS'(3)) >> 2); // Integer division for ceil

  logic [CT_BITS-1:0] initial_cnt [TOTAL_PES];
  logic [CT_BITS-1:0] curr_cnt    [TOTAL_PES];
  logic               active      [TOTAL_PES];
  logic [N_BITS-1:0]  base_row    [TOTAL_PES];
  logic [N_BITS-1:0]  base_col    [TOTAL_PES];
  logic [CH_BITS-1:0] channels    [TOTAL_PES];

  // Base row and column could be stored in one register currently as the STA only computes one base row/col at a time
  // In the future, further pipelining of STA could mean that multiple base row/cols are active at a time, meaning each
  // PE would need its own base row/col register

  always_ff @(posedge clk) begin
    if (reset) begin
      for (int k = 0; k < TOTAL_PES; k++) begin
        initial_cnt[k] <= '0;
        curr_cnt   [k] <= '0;
        active     [k] <= 1'b0;
        base_row   [k] <= '0;
        base_col   [k] <= '0;
        channels   [k] <= '0;
      end
    end else begin
      // Countdown and retire each PE
      for (int k = 0; k < TOTAL_PES; k++) begin
        if (active[k]) begin
          if (curr_cnt[k] != 'b0) begin
            curr_cnt[k] <= curr_cnt[k] - 1;
          end else begin
            // finished this cycle → clear active state for the next cycle
            active[k] <= 1'b0;
          end
        end
      end

      if (input_valid) begin
        initial_cnt [PE00_FLAT_IDX] <= compute_cycles;
        curr_cnt    [PE00_FLAT_IDX] <= compute_cycles - 1; // curr_cnt is 0 on the last cycle
        active      [PE00_FLAT_IDX] <= 1'b1;
        base_row    [PE00_FLAT_IDX] <= pos_row;
        base_col    [PE00_FLAT_IDX] <= pos_col;
        channels    [PE00_FLAT_IDX] <= channel;
      end

      // Propagate down column 0 from above neighbor
      for (int i = 1; i < ROWS; i++) begin
        int flat_idx_curr_pe   = i * COLS + 0;
        int flat_idx_above_pe = (i-1) * COLS + 0;
        // when PE(i-1,0) has just started (curr_cnt was set to initial_cnt - 1 in the previous cycle)
        if (active[flat_idx_above_pe] && (curr_cnt[flat_idx_above_pe] == initial_cnt[flat_idx_above_pe] - 1'b1)) begin
          initial_cnt [flat_idx_curr_pe] <= initial_cnt[flat_idx_above_pe];
          curr_cnt    [flat_idx_curr_pe] <= initial_cnt[flat_idx_above_pe] - 1'b1;
          active      [flat_idx_curr_pe] <= 1'b1;
          base_row    [flat_idx_curr_pe] <= base_row[flat_idx_above_pe];
          base_col    [flat_idx_curr_pe] <= base_col[flat_idx_above_pe];
          channels    [flat_idx_curr_pe] <= channels[flat_idx_above_pe];
        end
      end

      // Propagate right in each row from left neighbor
      for (int i = 0; i < ROWS; i++) begin
        for (int j = 1; j < COLS; j++) begin
          int flat_idx_curr_pe = i * COLS + j;
          int flat_idx_left_pe = i * COLS + (j-1);
          // when PE(i,j-1) has just started
          if (active[flat_idx_left_pe] && (curr_cnt[flat_idx_left_pe] == initial_cnt[flat_idx_left_pe] - 1'b1)) begin
            initial_cnt [flat_idx_curr_pe] <= initial_cnt[flat_idx_left_pe];
            curr_cnt    [flat_idx_curr_pe] <= initial_cnt[flat_idx_left_pe] - 1'b1;
            active      [flat_idx_curr_pe] <= 1'b1;
            base_row    [flat_idx_curr_pe] <= base_row[flat_idx_left_pe];
            base_col    [flat_idx_curr_pe] <= base_col[flat_idx_left_pe];
            channels    [flat_idx_curr_pe] <= channels[flat_idx_left_pe];
          end
        end
      end
    end
  end

  // Output valid & coordinates when outputs are valid
  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      for (int j = 0; j < COLS; j++) begin
        int flat_idx = i * COLS + j;
        // Output is valid for the cycle where curr_cnt becomes 0
        out_valid[flat_idx] = active[flat_idx] && (curr_cnt[flat_idx] == '0);
        // Calculate absolute row/col for the output using offset from PE(0,0)
        out_row  [flat_idx] = base_row[flat_idx] + N_BITS'(i);
        out_col  [flat_idx] = base_col[flat_idx] + N_BITS'(j);
        out_channels[flat_idx] = channels[flat_idx];
      end
    end
  end

endmodule
