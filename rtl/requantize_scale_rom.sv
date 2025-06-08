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
