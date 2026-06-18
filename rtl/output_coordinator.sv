`include "sys_types.svh"

// ======================================================================================================
// OUTPUT COORDINATOR
// ======================================================================================================
// This module manages the output timing and spatial coordinate generation for the systolic tensor
// array (STA) in the TinyML accelerator. It ensures proper synchronization between STA completion
// and output generation, providing spatial coordinates for downstream processing.
//
// FUNCTIONALITY:
// - Waits for STA completion and idle signals before generating outputs
// - Generates spatial coordinates for each processing element output
// - Provides timing control for output validity and data collection
// - Manages state transitions between computation and output phases
// - Ensures proper synchronization with upstream and downstream modules
//
// OPERATION:
// - Monitors 'done' signal from layer controller and 'sta_idle' from systolic array
// - When both signals are asserted, generates output valid signals for one cycle
// - Calculates absolute spatial coordinates based on block position and PE location
// - Provides flat array outputs for easy interfacing with downstream modules
//
// COORDINATE GENERATION:
// - Uses base position (pos_row, pos_col) from controller
// - Adds PE-specific offsets to generate absolute coordinates
// - Outputs coordinates for each PE in the array (ROWS*COLS total)
// - Maintains spatial ordering for downstream processing
//
// TIMING CONTROL:
// - Single-cycle output valid pulse when conditions are met
// - Prevents reading stale data without proper reset
// - Idle signal indicates coordinator is ready for next computation
// - Stall capability for flow control integration
//
// INTEGRATION:
// - Used by sta_controller to manage STA output timing
// - Receives control signals from layer controller and STA
// - Outputs to requantize_controller for post-processing
// - Coordinates with tensor RAM for spatial data storage
//
// PARAMETERS:
// - ROWS: Number of PE rows (typically 4)
// - COLS: Number of PE columns (typically 4)
// - MAX_N: Maximum matrix dimension for coordinate calculations
// - N_BITS: Bit-width for coordinate representation
//
// STATE MACHINE:
// - IDLE: Waiting for computation completion
// - OUTPUT_VALID: Generating outputs for one cycle
// - READ_ARRAY: Prevents stale data reads
// ======================================================================================================

module output_coordinator #(
  parameter int ROWS    = 4                 // # of PE rows
  ,parameter int COLS    = 4                // # of PE columns
  ,parameter int MAX_N   = 64              // Max matrix dimension calculated
  ,parameter int N_BITS  = $clog2(MAX_N)  // Bits to hold mat_size & coords
)(
  input  logic                clk
  ,input  logic               reset
  ,input  logic               stall                    // Freeze all calculations
  ,input  logic [N_BITS-1:0]  pos_row                  // Block's base row
  ,input  logic [N_BITS-1:0]  pos_col                  // Block's base col
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
  logic read_array; // Never read stale data again without resetting

  // State machine to handle output coordination
  always_ff @(posedge clk) begin
    if (reset) begin
      output_valid_state <= 1'b0;
      read_array <= 1'b0;
    end else if (~stall) begin
      if (done & sta_idle & ~output_valid_state & ~read_array) begin
        // Both done and idle are asserted, start outputting valid data
        output_valid_state <= 1'b1;
        read_array <= 1'b1;
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
        // Output is valid when in the output valid state
        out_valid[flat_idx] = output_valid_state;
        // Calculate absolute row/col for the output using offset from base position
        out_row[flat_idx] = pos_row + N_BITS'(i);
        out_col[flat_idx] = pos_col + N_BITS'(j);
      end
    end
    // Output coordinator is idle when not in output valid state or when systolic array is not idle and done
    idle = ~output_valid_state & read_array;
  end

endmodule
