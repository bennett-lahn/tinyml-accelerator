`include "sys_types.svh"

// ======================================================================================================
// REQUANTIZE CONTROLLER
// ======================================================================================================
// This module manages the requantization and activation pipeline for the TinyML accelerator's systolic
// tensor array (STA) output processing. It converts 32-bit accumulator results from the STA into
// 8-bit quantized activations using layer-specific scaling parameters and applies ReLU6 activation.
//
// FUNCTIONALITY:
// - Receives 32-bit accumulator outputs from the STA via output coordinator
// - Buffers incoming data using array_output_buffer instances for each channel
// - Applies layer-specific requantization using shared scale ROM parameters
// - Performs ReLU6 activation with configurable bypass capability
// - Outputs 8-bit quantized activations with spatial coordinates
//
// ARCHITECTURE:
// - SA_N parallel channels, each with dedicated buffer and requantize/activate unit
// - Shared scale ROM provides layer-specific quantization parameters
// - Array output buffers handle streaming data with handshake protocol
// - Requantize/activate units perform the actual quantization and activation
//
// INTEGRATION:
// - Instantiated in sta_controller.sv between output_coordinator and maxpool_unit
// - Receives STA outputs after bias addition and accumulation
// - Outputs to maxpool_unit for spatial downsampling (when not bypassed)
// - Can bypass ReLU activation for specific layers (e.g., final dense layer)
//
// PARAMETERS:
// - SA_N: Number of parallel channels (systolic array columns)
// - NUM_LAYERS: Total number of layers in the model
// - MAX_N: Maximum matrix dimension for coordinate calculations
// - MULT_WIDTH/SHIFT_WIDTH: Bit-widths for quantization parameters
//
// QUANTIZATION:
// - Uses TFLite-compatible quantization with fixed-point multipliers
// - Supports layer-specific zero points (normal: -128, special: -16)
// - Implements ReLU6 clamping with configurable qmax values per layer
// - Handles both positive and negative shift operations
// ======================================================================================================

module requantize_controller #(
  parameter int SA_N               = 4   // # of buffer/quant pairs
  ,parameter int NUM_LAYERS        = 6   // # of layers in model
  ,parameter int MISC_LAYER_IDX    = -1  // index for misc scales
  ,parameter int SPECIAL_LAYER_IDX = 5   // layer that uses special zero-point
  ,parameter int MULT_WIDTH        = 32  // bit-width for multipliers
  ,parameter int SHIFT_WIDTH       = 6   // bit-width for shifts
  ,parameter int MAX_N             = 64  // Max rows/cols in buffer
  ,parameter int N_BITS            = $clog2(MAX_N)
)(
  input   logic                                  clk
  ,input  logic                                  reset

  // Current layer
  ,input  logic signed[$clog2(NUM_LAYERS+1)-1:0] layer_idx
  ,input  logic bypass_relu // High if RELU function should be bypassed

  // Write-side: parallel write ports to all buffers
  ,input  logic              in_valid  [SA_N*4]
  ,input  int32_t            in_output [SA_N*4]
  ,input  logic [N_BITS-1:0] in_row    [SA_N*4]
  ,input  logic [N_BITS-1:0] in_col    [SA_N*4]

  // Requantized & activated outputs
  ,output logic              idle
  ,output logic              out_valid [SA_N]
  ,output logic [N_BITS-1:0] out_row   [SA_N]
  ,output logic [N_BITS-1:0] out_col   [SA_N]
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
    for (genvar ch = 0; ch < SA_N; ch++) begin : ARRAY_BUFFER
      array_output_buffer #(
        .MAX_N  (MAX_N)
        ,.N_BITS (N_BITS)
        ,.NUM_WRITE_PORTS(SA_N)
        ,.MAX_BUFFER_ENTRIES(SA_N)
      ) array_buffer_inst (
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
    ,.MULT_WIDTH     (MULT_WIDTH)
    ,.SHIFT_WIDTH    (SHIFT_WIDTH)
  ) scale_rom_inst (
    .clk             (clk)
    ,.valid          (1'b1) // TODO: not great syle to have valid always on
    ,.layer_idx      (layer_idx)
    ,.input_mult_out (input_mult)
    ,.input_shift_out(input_shift)
  );

  // === Shared quantization parameters for all channels ===
  logic choose_zero_point;

  assign choose_zero_point = (layer_idx == ($clog2(NUM_LAYERS+1))'(SPECIAL_LAYER_IDX));

  // === ReLU6 Quantization Range LUT ===
  // Provides QMIN and QMAX values for each layer's ReLU6 activation
  logic signed [7:0] qmin_lut;
  logic signed [7:0] qmax_lut;

  // All ReLU6 values are correct in this LUT
  always_comb begin
    case (layer_idx)
      3: qmax_lut = -102; 
      4: qmax_lut = -103;
      default: begin qmax_lut = 127; end
    endcase
  end

  // === Requantize/Activate Units and Control ===
  genvar ch;
  generate
    for (ch = 0; ch < SA_N; ch++) begin : QUANTIZE_UNIT
      // Connect buffer output to quant unit
      requantize_activate_unit qa_inst (
        .acc(buf_out_output[ch])
        ,.quant_mult(input_mult)
        ,.shift(input_shift)
        ,.choose_zero_point(choose_zero_point)
        ,.bypass_relu(bypass_relu)
        ,.qmax_in(qmax_lut)
        ,.out(out_data[ch])
      );

      // This logic creates a single-cycle pulse for buf_consume
      // to prevent back-to-back false consumption of buffer entries.
      logic last_buf_consume;
      always_ff @(posedge clk) begin
        if (reset)
          last_buf_consume <= 1'b0;
        else
          last_buf_consume <= buf_consume[ch];
      end

      // Only consume buffer entry when buffer has valid output.
      // last_buf_consume ensures that consume is only pulsed for one cycle.
      assign buf_consume[ch] = buf_out_valid[ch] & ~last_buf_consume;

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
      idle &= ~out_valid[i];
    end
  end

endmodule
