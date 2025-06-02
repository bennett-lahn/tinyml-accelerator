`include "sys_types.svh"

// This module does not perfectly implement TFlite's requantization algorithm, but a more hardware friendly adaptation
// May will result in minor errors that can be compensated for with quantization aware training (or ignored)
// This module likely needs to be pipelined; multiply+shift+shift+shift is a lot for one cycle
// but it's fine for now.
module requantize_activate_unit (
  input   int32_t acc                         // int32 MAC result + bias
  ,input  int32_t quant_mult                  // fixed-point multiplier; quant_mult and shift are used to represent the flop scale value as integer
  ,input  logic signed [5:0] shift            // right‐shift amount (can be negative)
  ,input  logic choose_zero_point             // output zero‐point; 0 if normal, 1 if special (1 only used for last dense layer in model)
  ,input  logic bypass_relu                   // High if relu should be bypassed
  ,input  int8_t qmax_in                      // max int8 value for ReLU6 clamping
  ,output int8_t out                          // requantized + ReLU6 output
);

  localparam int32_t norm_zero_point = -128;
  localparam int32_t special_zero_point = -16;
  int32_t zero_point;

  assign zero_point = (choose_zero_point) ? special_zero_point : norm_zero_point;

  // MultiplyByQuantizedMultiplier: (acc * M) with rounding, then shift
  // - Multiplies the 32-bit accumulator by a fixed-point multiplier (M).
  // - Rounds to nearest with a 64→32 bit shift (adds 1<<30 then >>>31).
  // - Applies a further signed shift (shift) to match the combined scale ratio
  // - from TFLite's quantization parameters
  function int32_t MultiplyByQuantizedMultiplier(int32_t val, int32_t multiplier, logic signed [5:0] quant_shift);
    logic signed [63:0] prod;
    logic signed [63:0] rounded;
    int32_t tmp;
    begin
      // 1) Full 64-bit product
      prod = val * multiplier;
      // 2) Round to nearest, tie to +∞: add 1<<30 before top-bit shift
      rounded = prod + (64'sd1 << 30);
      // 3) Shift down by 31 to align with Q31 format
      tmp = int32_t'(rounded >>> 31);
      // 4) Apply additional right/left shift
      if (quant_shift > 0)
        MultiplyByQuantizedMultiplier = tmp >>> quant_shift;
      else
        MultiplyByQuantizedMultiplier = tmp <<< -quant_shift;
    end
  endfunction

  int32_t scaled;
  int32_t with_zp;
  int8_t  clamped; // Used to apply ReLU6 activation

  always_comb begin
    // Requantize accumulator
    scaled = MultiplyByQuantizedMultiplier(acc, quant_mult, shift);
    // Add output zero-point
    with_zp = scaled + ((choose_zero_point) ? special_zero_point : norm_zero_point);
    // Clamp to [QMIN, QMAX] (implements quantized ReLU6) or full int8 range if bypassing ReLU
    if (bypass_relu) begin
      // Bypass ReLU: clamp to full int8 range
      if      (with_zp < -128) clamped = int8_t'(-128);
      else if (with_zp > 127)  clamped = int8_t'(127);
      else                     clamped = with_zp[7:0];
    end else begin
      // Normal ReLU6: clamp to [qmin_in, qmax_in]
      if      (with_zp < zero_point) clamped = int8_t'(zero_point);
      else if (with_zp > int32_t'(qmax_in))    clamped = int8_t'(qmax_in);
      else                           clamped = with_zp[7:0];
    end  
    out = clamped;
  end

endmodule
