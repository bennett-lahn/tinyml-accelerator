`include "sys_types.svh"

module output_coordinator #(
  parameter int ROWS    = 4                 // # of PE rows
  ,parameter int COLS    = 4                // # of PE columns
  ,parameter int MAX_N   = 512              // Max matrix dimension calculated
  ,parameter int N_BITS  = $clog2(MAX_N+1)  // Bits to hold mat_size & coords
)(
  input  logic                clk
  ,input  logic               reset
  ,input  logic               stall                    // Freeze all calculations
  ,input  logic [N_BITS-1:0]  pos_row                  // Block's base row
  ,input  logic [N_BITS-1:0]  pos_col                  // Block's base col
  ,input  logic               pe_mask [ROWS*COLS]      // 1 if PE should be active, 0 if ignored
  ,input  logic               done                     // Signal from layer controller indicating computation is done
  ,input  logic               sta_idle                 // Signal from systolic array indicating it's idle

  ,output logic               idle                     // High if output coordinator is idle
  ,output logic               out_valid    [ROWS*COLS] // High if corresponding PE output is valid
  ,output logic [N_BITS-1:0]  out_row      [ROWS*COLS] // Row in channel for each PE
  ,output logic [N_BITS-1:0]  out_col      [ROWS*COLS] // Col in channel for each PE
);

  localparam int TOTAL_PES = ROWS * COLS;

  // State to track if we're currently outputting valid data
  logic output_valid_state;

  // State machine to handle output coordination
  always_ff @(posedge clk) begin
    if (reset) begin
      output_valid_state <= 1'b0;
    end else if (~stall) begin
      if (done && sta_idle && ~output_valid_state) begin
        // Both done and idle are asserted, start outputting valid data
        output_valid_state <= 1'b1;
      end else if (output_valid_state) begin
        // Clear the output valid state after one cycle
        output_valid_state <= 1'b0;
      end
    end
  end

  // Output logic
  always_comb begin
    for (int i = 0; i < ROWS; i++) begin
      for (int j = 0; j < COLS; j++) begin
        int flat_idx = i * COLS + j;
        // Output is valid when in the output valid state and PE is active
        out_valid[flat_idx] = output_valid_state && pe_mask[flat_idx];
        // Calculate absolute row/col for the output using offset from base position
        out_row[flat_idx] = pos_row + N_BITS'(i);
        out_col[flat_idx] = pos_col + N_BITS'(j);
      end
    end
    
    // Output coordinator is idle when not in output valid state
    idle = ~output_valid_state;
  end

endmodule
