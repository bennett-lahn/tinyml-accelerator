`include "sys_types.svh"

module requantize_controller #(
  parameter int SA_N               = 4   // # of buffer/quant pairs
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

  // Current layer
  ,input  logic signed[$clog2(NUM_LAYERS+1)-1:0] layer_idx

  // Write-side: parallel write ports to all buffers
  ,input  logic              in_valid  [SA_N*4]
  ,input  int32_t            in_output [SA_N*4]
  ,input  logic [N_BITS-1:0] in_row    [SA_N*4]
  ,input  logic [N_BITS-1:0] in_col    [SA_N*4]

  // Requantized & activated outputs
  ,output logic              idle
  ,output logic              out_valid [SA_N]
  ,output logic [N_BITS-1:0] out_row   [SA_N]
  ,output logic [$clog2(MAX_N)-1:0] out_col   [SA_N]
  ,output int8_t             out_data  [SA_N]
);

  // === Array Output Buffers ===
  // Each buffer outputs one value per cycle, with handshake
  logic              buf_idle        [SA_N];
  logic              buf_out_valid   [SA_N];
  int32_t            buf_out_output  [SA_N];
  logic [N_BITS-1:0] buf_out_row     [SA_N];
  logic [N_BITS-1:0] buf_out_col     [SA_N];
  logic              buf_consume     [SA_N];

  // Intermediate signals for connecting unpacked arrays to buffer instances
  logic              buf_in_valid   [SA_N][4];
  int32_t            buf_in_output  [SA_N][4];
  logic [N_BITS-1:0] buf_in_row     [SA_N][4];
  logic [N_BITS-1:0] buf_in_col     [SA_N][4];

  // Connect input arrays to intermediate signals
  always_comb begin
    for (int ch = 0; ch < SA_N; ch++) begin
      for (int port = 0; port < 4; port++) begin
        buf_in_valid[ch][port]  = in_valid[ch*4 + port];
        buf_in_output[ch][port] = in_output[ch*4 + port];
        buf_in_row[ch][port]    = in_row[ch*4 + port];
        buf_in_col[ch][port]    = in_col[ch*4 + port];
      end
    end
  end

  generate
    for (genvar ch = 0; ch < SA_N; ch++) begin : BUF
      array_output_buffer #(
        .MAX_N  (MAX_N)
        ,.N_BITS (N_BITS)
        ,.NUM_WRITE_PORTS(SA_N)
        ,.MAX_BUFFER_ENTRIES(SA_N)
      ) buf_inst (
        .clk          (clk)
        ,.reset       (reset)
        ,.in_valid    (buf_in_valid[ch])
        ,.in_output   (buf_in_output[ch])
        ,.in_row      (buf_in_row[ch])
        ,.in_col      (buf_in_col[ch])
        ,.idle        (buf_idle[ch])
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
    .NUM_LAYERS      (NUM_LAYERS)
    ,.MISC_LAYER_IDX (MISC_LAYER_IDX)
    ,.MULT_WIDTH     (MULT_WIDTH)
    ,.SHIFT_WIDTH    (SHIFT_WIDTH)
  ) scale_rom_inst (
    .clk             (clk)
    ,.valid          (1'b1) // Always valid: fetch scale for current layer TODO: not great syle to have valid always on
    ,.layer_idx      (layer_idx)
    ,.input_mult_out (input_mult)
    ,.input_shift_out(input_shift)
  );

  // === Shared quantization parameters for all channels ===
  logic choose_zero_point;

  assign choose_zero_point = (layer_idx == SPECIAL_LAYER_IDX);

  // === Requantize/Activate Units and Control ===
  generate
    for (ch = 0; ch < SA_N; ch++) begin : QUANT
      // Connect buffer output to quant unit
      requantize_activate_unit #(
        .QMIN(-128)
        ,.QMAX(127)
      ) qa_inst (
        .acc(buf_out_output[ch])
        ,.quant_mult(input_mult)
        ,.shift(input_shift)
        ,.choose_zero_point(choose_zero_point)
        ,.out(out_data[ch])
      );

      // Only consume buffer entry when scale/shift are valid and buffer has valid output
      assign buf_consume[ch] = buf_out_valid[ch];

      // Output valid when buffer output is valid
      assign out_valid[ch] = buf_out_valid[ch];
      assign out_row[ch] = buf_out_row[ch];
      assign out_col[ch] = buf_out_col[ch];
    end
  endgenerate

  // Requantize idle if all activation units and buffers are idle and have no valid output
  always_comb begin
    idle = 1'b1;
    for (int i = 0; i < SA_N; i++) begin
      idle &= buf_idle[i];
      idle &= out_valid[i];
    end
  end

endmodule
