module requantize_scale_rom #(
  parameter int NUM_LAYERS = 6
  ,parameter int MULT_WIDTH = 32                                 // Width of multiplier for quantizing
  ,parameter int SHIFT_WIDTH = 6                                 // Width of shift for quantizing 
) (
  input  logic clk
  ,input  logic valid                                            // High if read request is valid
  ,input  logic [$clog2(NUM_LAYERS)-1:0] layer_idx      // -1 for misc, 0..NUM_LAYERS-1 for layers
  ,output logic signed [MULT_WIDTH-1:0] input_mult_out           // Multiplier for the weight scale for selected channel
  ,output logic signed [SHIFT_WIDTH-1:0] input_shift_out         // Shift for the weight scale for selected channel
);

  // Compute total ROM depth for ROM size
  typedef int layer_addr_t [$clog2(NUM_LAYERS)];
  localparam layer_addr_t LAYER_ADDRS = compute_layer_addresses();
  // The +1 below is for last output scale (not necessary as last layer doesn't requantize but eliminates out of bounds risk)
  localparam int LAYER_TOTAL_ENTRIES = LAYER_ADDRS[NUM_LAYERS];
  localparam int DATA_WIDTH = MULT_WIDTH + SHIFT_WIDTH;

  // ROM storage
  logic [DATA_WIDTH-1:0] rom [0:LAYER];

  // Precompute layer start addresses
  function automatic layer_addr_t compute_layer_addresses();
    layer_addr_t layer_addresses;
    int accum = MISC_SCALES;
    for (int i = 0; i < NUM_LAYERS; i++) begin
      layer_addresses[i] = accum;
      accum += 1; // +1 for output scale
    end
  endfunction

  // Address calculation
  logic [$clog2(DEPTH)-1:0] addr;
  always_comb begin
    if (layer_idx == MISC_LAYER_IDX) begin
      addr = LAYER_ADDRS[layer_idx] + weight_idx; // Unused for current model, may be helpful for future models
    end else begin
      addr = LAYER_ADDRS[layer_idx] + weight_idx;
    end
  end

  // Synchronous read
  always_ff @(posedge clk) begin
    if (valid) begin
      {input_mult_out, input_shift_out} <= rom[LAYER_ADDRS[layer_idx+1]-1]; // Get output scale, always last item
    end else begin
      {input_mult_out, input_shift_out} <= 'b0;
    end
  end

  // Initialization
  // initial begin
  //   // $readmemh("quant_params.mem", rom);
  // end

endmodule
