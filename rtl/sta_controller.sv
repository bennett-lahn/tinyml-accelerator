`include "sys_types.svh"

module sta_controller #(
  parameter int MAX_N     = 64                  // Max matrix dimension for output_coordinator
  ,parameter int N_BITS = $clog2(MAX_N)
  ,parameter int MAX_NUM_CH  = 64                   // Max number of channels in a layer
  ,parameter int CH_BITS     = $clog2(MAX_NUM_CH+1) // Bits to hold channel number
  ,parameter int MAX_BYPASS_IDX = 64                // Max index value for bypass mode (fully connected layers)
  ,parameter int BYPASS_IDX_BITS = $clog2(MAX_BYPASS_IDX) // Bits to hold bypass index
)(
  input  logic clk
  ,input  logic reset
  ,input  logic reset_sta  // Separate reset for everything except output buffer
  ,input  logic stall
  ,input  logic bypass_maxpool
  ,input  logic bypass_relu
  
  // Bypass inputs - used when bypass_maxpool is asserted
  ,input  logic   bypass_valid                      // Valid signal for bypass value
  ,input  int32_t bypass_value                      // Single int32 value to process when bypassing
  ,input  logic [BYPASS_IDX_BITS-1:0] bypass_index // Index for bypass value (for fully connected layers)

  // Inputs for Output Coorditor (to start/define a block computation)
  ,input  logic [N_BITS-1:0]   controller_pos_row            // Base row for the output block
  ,input  logic [N_BITS-1:0]   controller_pos_col            // Base col for the output block
  ,input  logic                done                          // Signal from layer controller indicating computation is done

  // Inputs for requantization controller
  ,input  logic [2:0]          layer_idx                     // Index of curent layer for computation

  // Inputs for systolic array
  // Systolic array controller assumes all inputs are already properly buffered/delayed for correct computation
  // A matrix inputs: one 4-wide int8 vector per row, left inputs
  ,input  int8_t   A0 [SA_VECTOR_WIDTH]        // Row 0 of A
  ,input  int8_t   A1 [SA_VECTOR_WIDTH]        // Row 1 of A
  ,input  int8_t   A2 [SA_VECTOR_WIDTH]        // Row 2 of A
  ,input  int8_t   A3 [SA_VECTOR_WIDTH]        // Row 3 of A

  // B matrix inputs: one 4-wide int8 vector per column, top inputs
  ,input  int8_t   B0 [SA_VECTOR_WIDTH]        // Column 0 of B
  ,input  int8_t   B1 [SA_VECTOR_WIDTH]        // Column 1 of B
  ,input  int8_t   B2 [SA_VECTOR_WIDTH]        // Column 2 of B
  ,input  int8_t   B3 [SA_VECTOR_WIDTH]        // Column 3 of B

  ,input  logic                         load_bias                    // Single load_bias control signal
  ,input  int32_t                       bias_value                  // Single bias value to be used for all PEs

  ,output logic                       idle    
  ,output logic                       sta_idle
  ,output logic                       array_out_valid               // High if corresponding val/row/col is valid
  ,output logic [7:0]                 array_val_out                 // 8-bit value out (changed from 128-bit)
  ,output logic [N_BITS-1:0]          array_row_out                 // Row for corresponding value
  ,output logic [N_BITS-1:0]          array_col_out                 // Column for corresponding value
  ,output logic [BYPASS_IDX_BITS-1:0] array_index_out               // Index for corresponding value (bypass mode)
);

  // Internal dimensions for the 4x4 Systolic Array
  localparam int SA_N              = 4;  // N x N Systolic Array size
  localparam int SA_VECTOR_WIDTH   = 4;  // Vector width for STA inputs (A, B)
  localparam int SA_TILE_SIZE      = 1;  // Tile size for STA (passed to STA instance)

  // Outputs from the controller - Flattened 1D Unpacked Arrays (per PE)
  localparam int CTRL_N_BITS = $clog2(MAX_N); // For coordinate width
  localparam int TOTAL_PES = SA_N * SA_N;          // Total PEs in the SA_N x SA_N array

  // Intermediate 2D arrays for unpacked inputs to STA (internal representation)

  logic   unpacked_load_bias_input[SA_N][SA_N];
  int32_t unpacked_bias_input     [SA_N][SA_N];

  // Internal wires connecting STA and Output Coordinator
  int32_t sta_C_outputs [SA_N][SA_N]; // STA still outputs in 2D internally

  // Output Coordinator outputs
  logic                          oc_idle;
  logic                          oc_out_valid_flat_internal [TOTAL_PES]; 
  logic [CTRL_N_BITS-1:0]        oc_out_row_flat_internal   [TOTAL_PES]; 
  logic [CTRL_N_BITS-1:0]        oc_out_col_flat_internal   [TOTAL_PES]; 
  logic [CH_BITS-1:0]            oc_out_chnnl_flat_internal [TOTAL_PES];

  // Unpack flattened 1D inputs into intermediate 2D arrays
  always_comb begin
    for (int i = 0; i < SA_N; i++) begin // Corresponds to STA row for A, controls; STA column for B
      // Replicate single bias control and value across all PEs
      for (int j = 0; j < SA_N; j++) begin 
        unpacked_load_bias_input[i][j] = load_bias;
        unpacked_bias_input[i][j]      = bias_value;
      end
    end
  end

  // Signal from systolic array indicating it's idle
  logic sta_idle_internal;

  assign sta_idle = sta_idle_internal;
  // Systolic array processes convolution inputs from exterior input buffers
  systolic_tensor_array #(
    .N(SA_N), 
    .TILE_SIZE(SA_TILE_SIZE),
    .VECTOR_WIDTH(SA_VECTOR_WIDTH)
  ) sta_unit (
    .clk(clk)
    ,.reset(reset|reset_sta)
    ,.stall(stall)
    ,.A0(A0)
    ,.A1(A1)
    ,.A2(A2)
    ,.A3(A3)
    ,.B0(B0)
    ,.B1(B1)
    ,.B2(B2)
    ,.B3(B3)
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
    ,.sta_idle(sta_idle_internal)
  );

  // Output coordinator waits for done and idle signals, then outputs all PE results
  output_coordinator #(
    .ROWS(SA_N)    
    ,.COLS(SA_N)    
    ,.MAX_N(MAX_N)
  ) oc_unit (
    .clk(clk)
    ,.reset(reset|reset_sta)
    ,.stall(stall) 
    ,.pos_row(controller_pos_row)
    ,.pos_col(controller_pos_col)
    ,.done(done)
    ,.sta_idle(sta_idle)
    ,.idle(oc_idle)
    ,.out_valid(oc_out_valid_flat_internal) 
    ,.out_row(oc_out_row_flat_internal)   
    ,.out_col(oc_out_col_flat_internal)    
  );

  // Flattened array of int32_t to be quantized by requant controller when valid
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

  logic                       requant_idle;             // High if requant unit is idle
  logic                       maxpool_idle;             // High if maxpool unit is idle
  logic                       requant_out_valid [SA_N]; // High if corresponding val/row/col is valid
  int8_t                      requant_val_out   [SA_N]; // Value out of each requant unit
  logic [$clog2(MAX_N)-1:0] requant_row_out   [SA_N]; // Row for corresponding value
  logic [$clog2(MAX_N)-1:0] requant_col_out   [SA_N]; // Column for corresponding value
  
  logic                       maxpool_out_valid;
  int8_t                      maxpool_val_out;
  logic [$clog2(MAX_N)-1:0] maxpool_row_out;
  logic [$clog2(MAX_N)-1:0] maxpool_col_out;

  // Bypass mode signals for requantize controller
  logic                     bypass_requant_valid_flat [SA_N*SA_N];
  int32_t                   bypass_requant_val_flat   [SA_N*SA_N]; 
  logic [CTRL_N_BITS-1:0]   bypass_requant_row_flat   [SA_N*SA_N];
  logic [CTRL_N_BITS-1:0]   bypass_requant_col_flat   [SA_N*SA_N];
  
  // Muxed inputs to requantize controller
  logic                     requant_in_valid_flat [SA_N*SA_N];
  int32_t                   requant_in_val_flat   [SA_N*SA_N];
  logic [CTRL_N_BITS-1:0]   requant_in_row_flat   [SA_N*SA_N];
  logic [CTRL_N_BITS-1:0]   requant_in_col_flat   [SA_N*SA_N];

  // Generate bypass signals - only first element is valid when bypassing
  always_comb begin
    for (int i = 0; i < SA_N*SA_N; i++) begin
      if (i == 0) begin
        // First element gets the bypass data
        bypass_requant_valid_flat[i] = bypass_valid;
        bypass_requant_val_flat[i]   = bypass_value;
        bypass_requant_row_flat[i]   = bypass_index;
        bypass_requant_col_flat[i]   = '0;
      end else begin
        // All other elements are invalid
        bypass_requant_valid_flat[i] = 1'b0;
        bypass_requant_val_flat[i]   = 32'd0;
        bypass_requant_row_flat[i]   = '0;
        bypass_requant_col_flat[i]   = '0;
      end
    end
  end

  // Mux between normal STA flow and bypass mode
  always_comb begin
    for (int i = 0; i < SA_N*SA_N; i++) begin
      if (bypass_maxpool) begin
        requant_in_valid_flat[i] = bypass_requant_valid_flat[i];
        requant_in_val_flat[i]   = bypass_requant_val_flat[i];
        requant_in_row_flat[i]   = bypass_requant_row_flat[i];
        requant_in_col_flat[i]   = bypass_requant_col_flat[i];
      end else begin
        requant_in_valid_flat[i] = oc_out_valid_flat_internal[i];
        requant_in_val_flat[i]   = requant_array_val[i];
        requant_in_row_flat[i]   = oc_out_row_flat_internal[i];
        requant_in_col_flat[i]   = oc_out_col_flat_internal[i];
      end
    end
  end

  // Reads input from STA as they become valid and requantizes them, 1 value per column of STA at a time
  // Supports SA_N simultaneous reads from each STA column per cycle, but only one requant operation per cycle 
  requantize_controller #(
    .MAX_N(MAX_N)
    ,.SA_N(SA_N)
  ) requant_unit (
    .clk(clk)
    ,.reset(reset)
    ,.bypass_relu(bypass_relu)
    ,.layer_idx(layer_idx)
    ,.in_valid(requant_in_valid_flat)
    ,.in_output(requant_in_val_flat)
    ,.in_row(requant_in_row_flat)
    ,.in_col(requant_in_col_flat)
    ,.idle(requant_idle)
    ,.out_valid(requant_out_valid)
    ,.out_row(requant_row_out)
    ,.out_col(requant_col_out)
    ,.out_data(requant_val_out)
  );

  // Requantized pixel value are stored in an SA_NxSA_N buffer, where max pooling is applied as values become available
  maxpool_unit #(
    .SA_N(SA_N)
    ,.MAX_N(MAX_N)
  ) pool_unit (
    .clk(clk)
    ,.reset(reset)
    ,.pos_row(controller_pos_row)
    ,.pos_col(controller_pos_col)
    ,.in_valid(requant_out_valid)
    ,.in_row(requant_row_out)
    ,.in_col(requant_col_out)
    ,.in_data(requant_val_out)
    ,.idle(maxpool_idle)
    ,.out_valid(maxpool_out_valid)
    ,.out_row(maxpool_row_out)
    ,.out_col(maxpool_col_out)
    ,.out_data(maxpool_val_out)
  );

  // Streaming output logic - output values as they become available
  logic                       array_out_valid_reg;
  logic [7:0]                  array_val_out_reg;
  logic [$clog2(MAX_N)-1:0]   array_row_out_reg;
  logic [$clog2(MAX_N)-1:0]   array_col_out_reg;
  logic [BYPASS_IDX_BITS-1:0] array_index_out_reg;

  always_ff @(posedge clk) begin
    if (reset) begin
      array_out_valid_reg <= 1'b0;
      array_val_out_reg <= 8'd0;
      array_row_out_reg <= '0;
      array_col_out_reg <= '0;
      array_index_out_reg <= '0;
    end else begin
      // Bypass mode: Output requantized values directly (skipping maxpool)
      if (bypass_maxpool) begin
        // Find the first valid requantized output this cycle
        array_out_valid_reg <= 1'b0;
        for (int i = SA_N-1; i >= 0; i--) begin
          if (requant_out_valid[i]) begin
            array_out_valid_reg <= 1'b1;
            array_val_out_reg <= requant_val_out[i];
            array_row_out_reg <= '0;  // Not used in bypass mode
            array_col_out_reg <= '0;  // Not used in bypass mode  
            array_index_out_reg <= requant_row_out[i][BYPASS_IDX_BITS-1:0]; // Use row output as index
          end
        end
      end else begin
        // Normal mode: Output max pooled values as they become available
        if (maxpool_out_valid) begin
          array_out_valid_reg <= 1'b1;
          array_val_out_reg <= maxpool_val_out;
          array_row_out_reg <= maxpool_row_out;
          array_col_out_reg <= maxpool_col_out;
          array_index_out_reg <= '0;  // Not used in normal mode
        end else begin
          array_out_valid_reg <= 1'b0;
        end
      end
    end
  end

  // Connect registered outputs to module outputs
  assign array_out_valid = array_out_valid_reg;
  assign array_val_out = array_val_out_reg;
  assign array_row_out = array_row_out_reg;
  assign array_col_out = array_col_out_reg;
  assign array_index_out = array_index_out_reg;

  // TODO: Ensure there is not a cycle gap where everything is idle because oc has finished sending requant data but
  // requant has not switched to not idle yet
  assign idle = sta_idle & oc_idle & requant_idle & maxpool_idle;

endmodule
