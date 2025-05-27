`include "sys_types.svh"

module sta_controller #(
  parameter int MAX_N     = 64                  // Max matrix dimension for output_coordinator
  ,parameter int N_BITS = $clog2(MAX_N)
  ,parameter int MAX_NUM_CH  = 64                   // Max number of channels in a layer
  ,parameter int CH_BITS     = $clog2(MAX_NUM_CH+1) // Bits to hold channel number
)(
  input  logic clk
  ,input  logic reset
  ,input  logic reset_sta  // Separate reset for everything except output buffer
  ,input  logic stall
  ,input  logic bypass_maxpool

  // Inputs for Output Coorditor (to start/define a block computation)
  ,input  logic [$clog2(MAX_N)-1:0]   controller_pos_row            // Base row for the output block
  ,input  logic [$clog2(MAX_N)-1:0]   controller_pos_col            // Base col for the output block
  ,input  logic                         done                          // Signal from layer controller indicating computation is done

  // Inputs for requantization controller
  ,input  logic [2:0]                        layer_idx                     // Index of new layer for computation

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

  ,output logic                         idle                          // High if STA complex is idle
  ,output logic                         array_out_valid               // High if corresponding val/row/col is valid
  ,output logic [127:0]                 array_val_out                 // Value out of max pool unit
  ,output logic [$clog2(MAX_N)-1:0]   array_row_out                 // Row for corresponding value
  ,output logic [$clog2(MAX_N)-1:0]   array_col_out                 // Column for corresponding value
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
  logic sta_idle;

  // Systolic array processes convolution inputs from exterior input buffers
  systolic_tensor_array #(
    .N(SA_N), 
    .TILE_SIZE(SA_TILE_SIZE),
    .VECTOR_WIDTH(SA_VECTOR_WIDTH)
  ) sta_unit (
    .clk(clk)
    ,.reset(reset)
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
    ,.sta_idle(sta_idle)
  );

  // Output coordinator waits for done and idle signals, then outputs all PE results
  output_coordinator #(
    .ROWS(SA_N)    
    ,.COLS(SA_N)    
    ,.MAX_N(MAX_N)
  ) oc_unit (
    .clk(clk)
    ,.reset(reset)
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

  // Reads input from STA as they become valid and requantizes them, 1 value per column of STA at a time
  // Supports SA_N simultaneous reads from each STA column per cycle, but only one requant operation per cycle 
  requantize_controller #(
    .MAX_N(MAX_N)
    ,.SA_N(SA_N)
  ) requant_unit (
    .clk(clk)
    ,.reset(reset_sta)
    ,.layer_idx(layer_idx)
    ,.in_valid(oc_out_valid_flat_internal)
    ,.in_output(requant_array_val)
    ,.in_row(oc_out_row_flat_internal)
    ,.in_col(oc_out_col_flat_internal)
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
    ,.reset(reset_sta)
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

  // Output buffer for max pooling: 4x4 buffer to hold four 2x2 chunks
  // Chunk arrangement: top-left, top-right, bottom-left, bottom-right
  logic output_buffer_valid [4][4];
  int8_t output_buffer [4][4];
  logic all_outputs_ready;
  logic [1:0] current_chunk;  // 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
  logic current_chunk_complete;
  
  // Registered outpu
  logic array_out_valid_reg;
  logic [127:0] array_val_out_reg;
  logic [$clog2(MAX_N)-1:0] array_row_out_reg;
  logic [$clog2(MAX_N)-1:0] array_col_out_reg;

  logic [N_BITS-1:0] rel_col;
  logic [N_BITS-1:0] rel_row;
  logic [N_BITS-1:0] base_row;
  logic [N_BITS-1:0] base_col;

  always_comb begin
    // Max pool mode: 2x2 chunks fill 4x4 buffer in specific order
    // Determine target position in 4x4 buffer based on current chunk
    case (current_chunk)
      2'b00: begin base_row = 0; base_col = 0; end  // Top-left
      2'b01: begin base_row = 0; base_col = 2; end  // Top-right  
      2'b10: begin base_row = 2; base_col = 0; end  // Bottom-left
      2'b11: begin base_row = 2; base_col = 2; end  // Bottom-right
    endcase

    rel_row = maxpool_row_out - controller_pos_row;
    rel_col = maxpool_col_out - controller_pos_col;
  end

  // Collect results into buffer until all outputs are ready, then output as 128-bit vector
  always_ff @(posedge clk) begin
    if (reset) begin
      // Full reset: reset everything including output buffer
      for (int i = 0; i < 4; i++) begin
        for (int j = 0; j < 4; j++) begin
          output_buffer_valid[i][j] <= 1'b0;
          output_buffer[i][j] <= 8'd0;
        end
      end
      array_out_valid_reg <= 1'b0;
      array_val_out_reg <= '0;
      array_row_out_reg <= '0;
      array_col_out_reg <= '0;
      current_chunk <= 2'b00;
    end else if (array_out_valid_reg) begin  
      // Clear all buffer valid bits when we output the complete tile
        for (int i = 0; i < 4; i++) begin
          for (int j = 0; j < 4; j++) begin
            output_buffer_valid[i][j] <= 1'b0;
          end
        end
        current_chunk <= 2'b00;
    // Update buffer with new data
    end else if (bypass_maxpool) begin
        // Direct mode: Requantized output goes directly to 4x4 buffer
        for (int i = 0; i < SA_N; i++) begin
          for (int j = 0; j < SA_N; j++) begin
            if (requant_out_valid[i] && requant_col_out[i] == N_BITS'(controller_pos_col + j)) begin
              output_buffer_valid[i][j] <= 1'b1;
              output_buffer[i][j] <= requant_val_out[i];
            end
          end
        end
      end else begin
        
        // Max pool outputs 2x2, map to appropriate chunk in 4x4 buffer
        if (maxpool_out_valid) begin
          if (rel_row < 2 && rel_col < 2) begin
            output_buffer_valid[2'(base_row + rel_row)][2'(base_col + rel_col)] <= 1'b1;
            output_buffer[2'(base_row + rel_row)][2'(base_col + rel_col)] <= maxpool_val_out;
          end
        end
        // Advance to next chunk when current chunk is complete
        if (current_chunk_complete && current_chunk != 2'b11) begin
          current_chunk <= current_chunk + 1;
        end
      end
      
      // Output assignment when all outputs are ready
      if (all_outputs_ready && !array_out_valid_reg) begin
        array_out_valid_reg <= 1'b1;
        // Pack all 16 outputs into 128-bit vector with MSB being first chunk
        // Chunk order: top-left, top-right, bottom-left, bottom-right
        // Within each chunk: [0][0], [0][1], [1][0], [1][1]
        array_val_out_reg <= {
          // Chunk 1 (top-left): bits [127:96]
          output_buffer[0][0], output_buffer[0][1], output_buffer[1][0], output_buffer[1][1],
          // Chunk 2 (top-right): bits [95:64]  
          output_buffer[0][2], output_buffer[0][3], output_buffer[1][2], output_buffer[1][3],
          // Chunk 3 (bottom-left): bits [63:32]
          output_buffer[2][0], output_buffer[2][1], output_buffer[3][0], output_buffer[3][1],
          // Chunk 4 (bottom-right): bits [31:0]
          output_buffer[2][2], output_buffer[2][3], output_buffer[3][2], output_buffer[3][3]
        };
        array_row_out_reg <= controller_pos_row;
        array_col_out_reg <= controller_pos_col;
      end else if (!all_outputs_ready) begin
        array_out_valid_reg <= 1'b0;
      end
    end

  // Combinational logic to check if current chunk is complete
  always_comb begin        
    current_chunk_complete = 1'b1;
    
    // Check if all positions in current 2x2 chunk are valid
    for (int i = 0; i < 2; i++) begin
      for (int j = 0; j < 2; j++) begin
        current_chunk_complete &= output_buffer_valid[2'(base_row + 6'(i))][2'(base_col + 6'(j))];
      end
    end
  end

  // Combinational logic detects when all outputs are ready
  always_comb begin
    if (bypass_maxpool) begin
      // Direct mode: check based on PE mask for 4x4 systolic array
      all_outputs_ready = 1'b1;
      for (int i = 0; i < SA_N; i++) begin
        for (int j = 0; j < SA_N; j++) begin
          int pe_idx = i * SA_N + j;  // Calculate flat PE index
          all_outputs_ready &= output_buffer_valid[i][j];
        end
      end
    end else begin
      // Max pool mode: check if all 4 chunks (2x2 each) are ready
      all_outputs_ready = 1'b1;
      for (int i = 0; i < 4; i++) begin
        for (int j = 0; j < 4; j++) begin
          all_outputs_ready &= output_buffer_valid[i][j];
        end
      end
    end
  end
  
  // Connect registered outputs to module outputs
  assign array_out_valid = array_out_valid_reg;
  assign array_val_out = array_val_out_reg;
  assign array_row_out = array_row_out_reg;
  assign array_col_out = array_col_out_reg;

  // TODO: Ensure there is not a cycle gap where everything is idle because oc has finished sending requant data but
  // requant has not switched to not idle yet
  assign idle = sta_idle & oc_idle & requant_idle & maxpool_idle;

endmodule
