`include "sys_types.svh"

module sta_controller #(
  parameter int OC_MAX_N = 512              // Max matrix dimension for output_coordinator
  ,parameter int NUM_CH  = 64               // Max number of channels in a layer
  ,parameter int CH_BITS = $clog2(NUM_CH+1) // Bits to hold channel number
)(
  input  logic clk
  ,input  logic reset
  ,input  logic stall

  // Inputs for Output Coordinator (to start/define a block computation)
  ,input  logic [$clog2(OC_MAX_N+1)-1:0] controller_mat_size           // Size of the current matrix operation
  ,input  logic                          controller_input_valid        // Signals start of a new computation
  ,input  logic [$clog2(OC_MAX_N+1)-1:0] controller_pos_row            // Base row for the output block
  ,input  logic [$clog2(OC_MAX_N+1)-1:0] controller_pos_col            // Base col for the output block
  ,input  logic [$clog2(NUM_CH+1)-1:0]   controller_channel            // Layer channel for the output block

  // Inputs for requantization controller
  ,input  logic                          layer_valid                   // High if new layer signal is valid
  ,input  logic                          layer_idx                     // Index of new layer for computation

  // Inputs for systolic array
  ,input  int8_t  A_input_rows [SA_N*SA_VECTOR_WIDTH]                  // Flattened A matrix (row-major)
  ,input  int8_t  B_input_cols [SA_N*SA_VECTOR_WIDTH]                  // Flattened B matrix (col-major from STA perspective)
  ,input  logic                          load_sum_per_row  [SA_N*SA_N] // Flattened load_sum controls (row-major)
  ,input  logic                          load_bias_per_row [SA_N*SA_N] // Flattened load_bias controls (row-major)
  ,input  int32_t                        bias_per_row      [SA_N*SA_N] // Flattened bias values (row-major)

  ,output logic                          array_out_valid               // High if corresponding val/row/col is valid
  ,output int32_t                        array_val_out                 // Value out of each requant unit
  ,output logic [$clog2(OC_MAX_N+1)-1:0] array_row_out                 // Row for corresponding value
  ,output logic [$clog2(OC_MAX_N+1)-1:0] array_col_out                 // Column for corresponding value
);

  // Internal dimensions for the 4x4 Systolic Array
  localparam int SA_N              = 4;  // N x N Systolic Array size
  localparam int SA_VECTOR_WIDTH   = 4;  // Vector width for STA inputs (A, B)
  localparam int SA_TILE_SIZE      = 1;  // Tile size for STA (passed to STA instance)

  // Outputs from the controller - Flattened 1D Unpacked Arrays (per PE)
  localparam int CTRL_N_BITS = $clog2(OC_MAX_N+1); // For coordinate width
  localparam int TOTAL_PES = SA_N * SA_N;          // Total PEs in the SA_N x SA_N array

  // Intermediate 2D arrays for unpacked inputs to STA (internal representation)
  int8_t  unpacked_A_input [SA_N][SA_VECTOR_WIDTH];
  int8_t  unpacked_B_input [SA_N][SA_VECTOR_WIDTH]; 
  logic   unpacked_load_sum_input [SA_N][SA_N];
  logic   unpacked_load_bias_input[SA_N][SA_N];
  int32_t unpacked_bias_input     [SA_N][SA_N];

  // Internal wires connecting STA and Output Coordinator
  int32_t sta_C_outputs [SA_N][SA_N]; // STA still outputs in 2D internally

  // Output Coordinator outputs (flattened, as before)
  logic                          oc_out_valid_flat_internal [TOTAL_PES]; 
  logic [CTRL_N_BITS-1:0]        oc_out_row_flat_internal   [TOTAL_PES]; 
  logic [CTRL_N_BITS-1:0]        oc_out_col_flat_internal   [TOTAL_PES]; 
  logic [CH_BITS-1:0]            oc_out_chnnl_flat_internal [TOTAL_PES];

  // Unpack flattened 1D inputs into intermediate 2D arrays
  always_comb begin
    for (int i = 0; i < SA_N; i++) begin // Corresponds to STA row for A, controls; STA column for B
      // Unpack A_input_rows (row-major flattened)
      // unpacked_A_input[i] is the i-th row vector for the STA
      for (int k = 0; k < SA_VECTOR_WIDTH; k++) begin
        unpacked_A_input[i][k] = A_input_rows[i * SA_VECTOR_WIDTH + k];
      end

      // Unpack B_input_cols (conceptually column-major flattened for STA)
      // unpacked_B_input[i] is the i-th column vector for the STA (B0, B1, etc.)
      for (int k = 0; k < SA_VECTOR_WIDTH; k++) begin
        unpacked_B_input[i][k] = B_input_cols[i * SA_VECTOR_WIDTH + k];
      end
      
      // Unpack control signals and bias (row-major flattened)
      // unpacked_load_sum_input[i][j] is for PE at row i, col j
      for (int j = 0; j < SA_N; j++) begin 
        unpacked_load_sum_input[i][j]  = load_sum_per_row[i * SA_N + j];
        unpacked_load_bias_input[i][j] = load_bias_per_row[i * SA_N + j];
        unpacked_bias_input[i][j]      = bias_per_row[i * SA_N + j];
      end
    end
  end

  // Systolic array processes convolution inputs from exterior input buffers
  systolic_tensor_array #(
    .N(SA_N), 
    .TILE_SIZE(SA_TILE_SIZE),
    .VECTOR_WIDTH(SA_VECTOR_WIDTH)
  ) sta_unit (
    .clk(clk)
    ,.reset(reset)
    ,.stall(stall)
    ,.A0(unpacked_A_input[0])
    ,.A1(unpacked_A_input[1])
    ,.A2(unpacked_A_input[2])
    ,.A3(unpacked_A_input[3])
    ,.B0(unpacked_B_input[0]) // unpacked_B_input[0] is the vector for B0 (col 0)
    ,.B1(unpacked_B_input[1]) // unpacked_B_input[1] is the vector for B1 (col 1)
    ,.B2(unpacked_B_input[2])
    ,.B3(unpacked_B_input[3])
    ,.load_sum0(unpacked_load_sum_input[0])
    ,.load_sum1(unpacked_load_sum_input[1])
    ,.load_sum2(unpacked_load_sum_input[2])
    ,.load_sum3(unpacked_load_sum_input[3])
    ,.load_bias0(unpacked_load_bias_input[0])
    ,.load_bias1(unpacked_load_bias_input[1])
    ,.load_bias2(unpacked_load_bias_input[2])
    ,.load_bias3(unpacked_load_bias_input[3])
    ,.bias0(unpacked_bias_input[0])
    ,.bias1(unpacked_bias_input[1])
    ,.bias2(unpacked_bias_input[2])
    ,.bias3(unpacked_bias_input[3])
    ,.C0(sta_C_outputs[0])
    ,.C1(sta_C_outputs[1])
    ,.C2(sta_C_outputs[2])
    ,.C3(sta_C_outputs[3])
  );

  // Output coordinator tracks compute time of PEs, triggers accumulator value reads when compute is finished
  output_coordinator #(
    .ROWS(SA_N)    
    ,.COLS(SA_N)    
    ,.MAX_N(OC_MAX_N)
    ,.NUM_CH(NUM_CH)
    ,.CH_BITS(CH_BITS)
  ) oc_unit (
    .clk(clk)
    ,.reset(reset)
    ,.mat_size(controller_mat_size)
    ,.input_valid(controller_input_valid)
    ,.stall(stall) 
    ,.pos_row(controller_pos_row)
    ,.pos_col(controller_pos_col)
    ,.channel(controller_channel)
    ,.out_valid(oc_out_valid_flat_internal) 
    ,.out_row(oc_out_row_flat_internal)   
    ,.out_col(oc_out_col_flat_internal)    
    ,.out_channels(oc_out_chnnl_flat_internal)
  );

  int32_t requant_array_val [SA_N*SA_N];

  // Generate controller outputs by assigning directly to the flattened 1D unpacked output arrays
  always_comb begin
    for (int i = 0; i < SA_N; i++) begin    // Iterating through conceptual rows of PEs
      for (int j = 0; j < SA_N; j++) begin  // Iterating through conceptual columns of PEs
        int flat_idx = i * SA_N + j;        // Calculate the flat index for this PE
        requant_array_val[flat_idx] = sta_C_outputs[i][j];
      end
    end
  end

  logic                          requant_out_valid [SA_N]; // High if corresponding val/row/col is valid
  int32_t                        requant_val_out   [SA_N]; // Value out of each requant unit
  logic [$clog2(OC_MAX_N+1)-1:0] requant_row_out   [SA_N]; // Row for corresponding value
  logic [$clog2(OC_MAX_N+1)-1:0] requant_col_out   [SA_N]; // Column for corresponding value

  // Reads input from STA as they become valid and requantizes them, 1 value per column of STA at a time
  // Supports SA_N simultaneous reads from each STA column per cycle, but only one requant operation per cycle 
  requantize_controller #(
    .MAX_N(OC_MAX_N)
    ,.SA_N(SA_N)
  ) requant_unit (
    .clk(clk)
    ,.reset(reset)
    ,.layer_valid(layer_valid)
    ,.layer_idx(layer_idx)
    ,.in_valid(oc_out_valid_flat_internal)
    ,.in_output(requant_array_val)
    ,.in_row(oc_out_row_flat_internal)
    ,.in_col(oc_out_col_flat_internal)
    ,.out_valid(requant_out_valid)
    ,.out_row(requant_row_out)
    ,.out_col(requant_col_out)
    ,.out_data(array_val_out)
  );

  // Requantized pixel value are stored in an SA_NxSA_N buffer, where max pooling is applied as values become available
  maxpool_unit #(
    .SA_N(SA_N)
    ,.MAX_N(OC_MAX_N)
  ) pool_unit (
    .clk(clk)
    ,.reset(reset)
    ,.pos_row(pos_row)
    ,.pos_col(pos_col)
    ,.in_valid(requant_out_valid)
    ,.in_row(requant_row_out)
    ,.in_col(requant_col_out)
    ,.in_data(array_val_out)
    ,.out_valid(array_out_valid)
    ,.out_row(array_row_out)
    ,.out_col(array_col_out)
    ,.out_data(array_col_out)
  );

endmodule
