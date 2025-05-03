// Module: weight_rom
// Description: Parameterizable synchronous ROM for CNN layer weights.
//              Stores weights for multiple layers with configurable kernel sizes
//              and channel counts. Supports single-cycle read access.

// TODO: Should ROMs be expected to return data in one cycle?
module weight_rom #(
    // Number of CNN layers
    parameter int NUM_LAYERS,
    
    // Array: Convolution kernel size (N x N) for each layer
    parameter int LAYER_MATRIX_SIZES [NUM_LAYERS-1:0],
    
    // Array: Input channels per layer (determines # of kernels per layer)
    parameter int LAYER_CHANNELS [NUM_LAYERS-1:0],
    
    // Bit width of each weight value
    parameter int WEIGHT_WIDTH = 8,
    
    // Packed array of all weight values in layer-major order:
    // Format: [Layer0_Kernel0_Channel0_W0, Layer0_Kernel0_Channel0_W1, ...,
    //          Layer0_Kernel0_Channel0_W(N²-1), Layer0_Kernel0_Channel1_W0, ...]
    parameter logic [WEIGHT_WIDTH-1:0] WEIGHT_VALUES []
)(
    // Clock input for synchronous read
    input  logic clk,
    
    // Layer selection (0 to NUM_LAYERS-1)
    input  logic [$clog2(NUM_LAYERS)-1:0] layer_sel,
    
    // Address within selected layer's weight memory space:
    // Address range per layer: [0 to (matrix_size² * channels) - 1]
    input  logic [$clog2(LAYER_MATRIX_SIZES[0]**2 * LAYER_CHANNELS[0])-1:0] addr,
    
    // Output: Weight value at specified layer and address
    output logic [WEIGHT_WIDTH-1:0] weight_out
);

// Function: calc_layer_offsets
// Description: Computes starting ROM indices for each layer's weights.
//              Returns array where offsets[i] = starting index of layer i
// Arguments:
//   layer_sizes - Kernel dimensions for each layer
//   channels    - Input channels for each layer
function automatic int calc_layer_offsets(input int layer_sizes[], input int channels[]);
    automatic int offsets[NUM_LAYERS];
    automatic int cumulative = 0;
  
    offsets[0] = 0;  // First layer starts at index 0
    for (int i = 1; i < NUM_LAYERS; i++) begin
        // Cumulative sum of weights from previous layers:
        // (kernel_size²) * channels for layer i-1
        cumulative += layer_sizes[i-1] ** 2 * channels[i-1];
        offsets[i] = cumulative;
    end
    return offsets;
endfunction

// Local Parameters:
//   LAYER_OFFSETS - Starting indices for each layer in the flat ROM array
//   TOTAL_WEIGHTS - Total number of weights across all layers
localparam int LAYER_OFFSETS [NUM_LAYERS] = calc_layer_offsets(
    LAYER_MATRIX_SIZES, 
    LAYER_CHANNELS
);
localparam int TOTAL_WEIGHTS = LAYER_OFFSETS[NUM_LAYERS-1] + 
                              (LAYER_MATRIX_SIZES[NUM_LAYERS-1]**2 * 
                              LAYER_CHANNELS[NUM_LAYERS-1]);

// ROM Storage: weight_rom
// Description: Block RAM storing all weights in layer-major order.
//              Initialized from WEIGHT_VALUES parameter during synthesis.
logic [WEIGHT_WIDTH-1:0] weight_rom [TOTAL_WEIGHTS];
initial begin
    foreach(weight_rom[i]) begin
        weight_rom[i] = WEIGHT_VALUES[i];
    end
end

// Address Calculation:
//   calc_addr - Combines layer offset with intra-layer address
//               Performs overflow protection (wraps to 0 if out of bounds)
logic [$clog2(TOTAL_WEIGHTS)-1:0] calc_addr;
always_comb begin
    // Base address for selected layer + intra-layer offset
    calc_addr = LAYER_OFFSETS[layer_sel] + addr;
    
    // Safety: Prevent out-of-bounds access
    if (calc_addr >= TOTAL_WEIGHTS)
        calc_addr = 0;  // Could implement error signaling instead
end

// Synchronous Read
always_ff @(posedge clk) begin
    weight_out <= weight_rom[calc_addr];
end

endmodule
