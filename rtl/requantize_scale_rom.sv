module requantize_scale_rom #(
  parameter int NUM_LAYERS = 28,
  parameter int MISC_SCALES = 10, // Other scales that don't fit within the convolution layers
  parameter int MULT_WIDTH = 32,  // Width of multiplier for quantizing
  parameter int SHIFT_WIDTH = 6,  // Width of shift for quantizing 
  // Array of weight counts per layer (e.g., '{8, 64, 128, ...})
  parameter int WEIGHT_SCALES_PER_LAYER [NUM_LAYERS] = '{default:128},
  parameter int MISC_LAYER_IDX = -1 // Use -1 in layer_idx to indicate misc scales
) (
  input  logic clk,
  input  logic valid, // High if read request is valid
  input  logic signed [$clog2(NUM_LAYERS+1)-1:0] layer_idx, // -1 for misc, 0..NUM_LAYERS-1 for layers
  input  logic [$clog2(max_weights_per_layer())-1:0] weight_idx, // 0..N for weights, N=output scale; WARNING: Must be < 9 for MISC_SCALES access
  output logic signed [MULT_WIDTH-1:0] mult_out,
  output logic signed [SHIFT_WIDTH-1:0] shift_out
);

  // Helper function: finds max weights per layer (for port sizing)
  function automatic int max_weights_per_layer();
    int maxval = 0;
    foreach (WEIGHT_SCALES_PER_LAYER[i])
      if (WEIGHT_SCALES_PER_LAYER[i] > maxval)
        maxval = WEIGHT_SCALES_PER_LAYER[i];
    return maxval + 1; // +1 for output scale
  endfunction

  // Compute total ROM depth for ROM size
  localparam int LAYER_ADDRS [NUM_LAYERS] = compute_layer_addresses();
  // +1 below for last output scale (not necessary as last layer doesn't requantize but eliminates out of bounds risk)
  localparam int LAYER_TOTAL_ENTRIES = LAYER_ADDRS[NUM_LAYERS-1] + WEIGHT_SCALES_PER_LAYER[NUM_LAYERS-1] + 1;
  localparam int DEPTH = MISC_SCALES + LAYER_TOTAL_ENTRIES;
  localparam int DATA_WIDTH = MULT_WIDTH + SHIFT_WIDTH;

  // ROM storage
  logic [DATA_WIDTH-1:0] rom [0:DEPTH-1];

  // Precompute layer start addresses
  function automatic int compute_layer_addresses [NUM_LAYERS]();
    int accum = MISC_SCALES;
    foreach (compute_layer_addresses[i]) begin
      compute_layer_addresses[i] = accum;
      accum += WEIGHT_SCALES_PER_LAYER[i] + 1; // +1 for output scale
    end
  endfunction

  // Address calculation
  logic [$clog2(DEPTH)-1:0] addr;
  always_comb begin
    if (layer_idx == MISC_LAYER_IDX) begin
      addr = weight_idx; // Misc. scales at front
    end else begin
      addr = LAYER_ADDRS[layer_idx] + weight_idx;
    end
  end

  // Synchronous read
  always_ff @(posedge clk) begin
    if (valid)
      {mult_out, shift_out} <= rom[addr];
    else
      {mult_out, shift_out} <= 'b0;
  end

  // Initialization
  initial begin
    $readmemh("quant_params.mem", rom);
  end

endmodule
