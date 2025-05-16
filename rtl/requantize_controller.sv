`include "sys_types.svh"

module requantize_controller #(
  parameter int NUM_CHANNELS      = 4    // # of buffer/quant pairs
  ,parameter int NUM_LAYERS        = 6   // # of layers in model
  ,parameter int MISC_LAYER_IDX    = -1  // index for misc scales
  ,parameter int SPECIAL_LAYER_IDX = 7   // layer that uses special zero-point
  ,parameter int MULT_WIDTH        = 32  // bit-width for multipliers
  ,parameter int SHIFT_WIDTH       = 6   // bit-width for shifts
  ,parameter int MAX_N             = 16  // Max rows/cols in buffer
  ,parameter int N_BITS            = $clog2(MAX_N)
)(
  input   logic                                  clk
  ,input  logic                                  reset

  // Layer selection handshake, pulse to latch new layer_idx
  ,input  logic                                  layer_valid
  ,input  logic signed[$clog2(NUM_LAYERS+1)-1:0] layer_idx

  // Write-side: parallel write ports to all buffers
  ,input  logic              in_valid  [NUM_CHANNELS*4]
  ,input  int32_t            in_output [NUM_CHANNELS*4]
  ,input  logic [N_BITS-1:0] in_row    [NUM_CHANNELS*4]
  ,input  logic [N_BITS-1:0] in_col    [NUM_CHANNELS*4]

  // Requantized & activated outputs
  ,output logic              out_valid [NUM_CHANNELS]
  ,output int8_t             out_data  [NUM_CHANNELS]
);

  // Registered layer index
  logic signed[$clog2(NUM_LAYERS+1)-1:0] layer_idx_reg;
  always_ff @(posedge clk or posedge reset) begin
    if (reset)                 layer_idx_reg <= '0;
    else if (layer_valid)      layer_idx_reg <= layer_idx;
  end

  // === Array Output Buffers ===
  // Each buffer outputs one value per cycle, with handshake
  logic              buf_out_valid   [NUM_CHANNELS];
  int32_t            buf_out_output  [NUM_CHANNELS];
  logic [N_BITS-1:0] buf_out_row     [NUM_CHANNELS];
  logic [N_BITS-1:0] buf_out_col     [NUM_CHANNELS];
  logic              buf_consume     [NUM_CHANNELS];

  generate
    for (genvar ch = 0; ch < NUM_CHANNELS; ch++) begin : BUF
      array_output_buffer #(
        .MAX_N  (MAX_N)
        ,.N_BITS (N_BITS)
      ) buf_inst (
        .clk         (clk)
        ,.reset       (reset)
        ,.in_valid    (in_valid[ch*4:(ch*4)+3])
        ,.in_output   (in_output[ch*4:(ch*4)+3])
        ,.in_row      (in_row[ch*4:(ch*4)+3])
        ,.in_col      (in_col[ch*4:(ch*4)+3])
        ,.out_valid   (buf_out_valid[ch])
        ,.out_output  (buf_out_output[ch])
        ,.out_row     (buf_out_row[ch])
        ,.out_col     (buf_out_col[ch])
        ,.out_consume (buf_consume[ch])
      );
    end
  endgenerate

  // === Shared Scale ROM ===
  // Only one ROM instance, shared for all channels
  logic signed [MULT_WIDTH-1:0]  input_mult;
  logic signed [SHIFT_WIDTH-1:0] input_shift;

  requantize_scale_rom #(
    .NUM_LAYERS     (NUM_LAYERS)
    ,.MISC_LAYER_IDX (MISC_LAYER_IDX)
    ,.MULT_WIDTH     (MULT_WIDTH)
    ,.SHIFT_WIDTH    (SHIFT_WIDTH)
  ) scale_rom_inst (
    .clk            (clk)
    ,.valid          (1'b1) // Always valid: fetch scale for current layer
    ,.layer_idx      (layer_idx_reg)
    ,.input_mult_out (input_mult)
    ,.input_shift_out(input_shift)
  );

  // === Shared quantization parameters for all channels ===
  logic choose_zero_point;

  assign choose_zero_point = (layer_idx_reg == SPECIAL_LAYER_IDX);

  // === Requantize/Activate Units and Control ===
  generate
    for (ch = 0; ch < NUM_CHANNELS; ch++) begin : QUANT
      // Connect buffer output to quant unit
      requantize_activate_unit #(
        .QMIN           (-128)
        ,.QMAX           (127)
      ) qa_inst (
        .acc            (buf_out_output[ch])
        ,.quant_mult     (input_mult)
        ,.shift          (input_shift)
        ,.choose_zero_point(choose_zero_point)
        ,.out            (out_data[ch])
      );

      // Only consume buffer entry when scale/shift are valid and buffer has valid output
      assign buf_consume[ch] = buf_out_valid[ch];

      // Output valid when buffer output is valid
      assign out_valid[ch] = buf_out_valid[ch];
    end
  endgenerate

endmodule
