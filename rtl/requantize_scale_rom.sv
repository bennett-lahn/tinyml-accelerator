// ======================================================================================================
// REQUANTIZE SCALE ROM
// ======================================================================================================
// This module provides layer-specific quantization parameters (multiplier and shift values) for the
// requantization process in the TinyML accelerator. It stores pre-computed scaling factors that
// convert between the internal 32-bit accumulator format and the 8-bit quantized output format.
//
// FUNCTIONALITY:
// - Stores quantization parameters for each layer in the model
// - Provides synchronous read access to multiplier and shift values
// - Supports initialization from external hex file
// - Includes bounds checking for layer index validation
// - Outputs concatenated multiplier and shift values per layer
//
// QUANTIZATION PARAMETERS:
// - Multiplier: Fixed-point scaling factor for requantization
// - Shift: Right/left shift amount for final scaling adjustment
// - Parameters are layer-specific and derived from TFLite model quantization
// - Stored as concatenated {shift, multiplier} in ROM
//
// INTEGRATION:
// - Used by requantize_controller to provide scaling parameters
// - Shared across all SA_N channels in the requantization pipeline
// - Parameters are loaded from quant_params.hex file during initialization
// - Supports NUM_LAYERS different layer configurations
//
// PARAMETERS:
// - NUM_LAYERS: Total number of layers requiring quantization parameters
// - MULT_WIDTH: Bit-width of the multiplier (default: 32)
// - SHIFT_WIDTH: Bit-width of the shift value (default: 6)
// - INIT_FILE: Path to hex file containing quantization parameters
//
// TIMING:
// - Synchronous read with clock edge
// - Valid signal controls read operations
// - Outputs are registered and stable for one clock cycle
// ======================================================================================================

module requantize_scale_rom #(
  parameter int NUM_LAYERS = 6
  ,parameter int MULT_WIDTH = 32                                 // Width of multiplier for quantizing
  ,parameter int SHIFT_WIDTH = 6                                 // Width of shift for quantizing 
  ,parameter INIT_FILE = "../fakemodel/quant_params.hex"

) (
  input  logic clk
  ,input  logic valid                                            // High if read request is valid
  ,input  logic [$clog2(NUM_LAYERS)-1:0] layer_idx      // -1 for misc, 0..NUM_LAYERS-1 for layers
  ,output logic signed [MULT_WIDTH-1:0] input_mult_out           // Multiplier for the weight scale for selected channel
  ,output logic signed [SHIFT_WIDTH-1:0] input_shift_out         // Shift for the weight scale for selected channel
);

  // ROM storage
  logic [MULT_WIDTH+SHIFT_WIDTH-1:0] rom [NUM_LAYERS];

  // Address bounds checking
  always_ff @(posedge clk) begin
    if (valid && layer_idx >= ($clog2(NUM_LAYERS))'(NUM_LAYERS)) begin
      $display("ERROR: requantize_scale_rom address out of bounds at time %0t! layer_idx=%d, max_valid_idx=%d", $time, layer_idx, NUM_LAYERS-1);
    end
  end

  // Synchronous read
  always_ff @(posedge clk) begin
    if (valid) begin
      {input_shift_out, input_mult_out} <= rom[layer_idx];
    end else begin
      {input_shift_out, input_mult_out} <= 'b0;
    end
  end

  // Initialization
  initial begin
        if (INIT_FILE != "") begin // Only initialize if a file is specified (using the new parameter)
            $display("requantize_scale_rom: Initializing ROM from file: %s", INIT_FILE);
            $readmemh(INIT_FILE, rom);
        end else begin
            $display("requantize_scale_rom: No INIT_FILE specified, ROM not initialized from file by $readmemh.");
        end
    end

endmodule
