// Parameterizable synchronous ROM for CNN layer weights
// Stores weights for multiple layers with configurable kernel sizes and channel counts
// Returns 16 weights at a time to support STA capacity
// Left shift 4 equivalent to multiplying by 16, used for hardware simplicity

// TODO: Note that method for calculating layer offsets, valid bit and total ROM size is untested
module weight_rom #(
    parameter int NUM_LAYERS                           // Number of CNN layers
    ,parameter int LAYER_MATRIX_SIZES [NUM_LAYERS-1:0] // Array: Convolution kernel size (N x N) for each layer
    ,parameter int LAYER_CHANNELS [NUM_LAYERS-1:0]     // Array: Input channels per layer (determines # of kernels per layer)
    ,parameter int WEIGHT_WIDTH = 8                    // Bit width of each weight value
    ,parameter int VECTOR_SIZE  = 4                    // Size of vector memory returns in number of weights
)(
    input  logic clk                                  // Clock input for synchronous read
    ,input  logic read_valid                           // High if read should return valid ROM contents 
    ,input  logic [$clog2(NUM_LAYERS)-1:0] layer_sel   // Layer selection (0 to NUM_LAYERS-1)

    // Address within selected layer's weight memory space, each address represents 16 weights
    // Address range per layer: [0 to ((matrix_size² * channels)/VECTOR_SIZE) - 1]



    // TODO

    ,input  logic [31:0] addr // TODO: HARDCODE SIZE BASED ON LARGEST LAYER WEIGHT COUNT LATER







    ,output logic valid_out [VECTOR_SIZE]               // Corresponding weight output is valid only if valid bit is high
    ,output logic [(WEIGHT_WIDTH*VECTOR_SIZE)-1:0] weight_out // Weight value at specified layer and address
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
      
        offsets[0] = 0; // First layer starts at index 0
        for (int i = 1; i < NUM_LAYERS; i++) begin
            // Cumulative sum of weights from previous layers:
            cumulative += $ceil((layer_sizes[i-1] ** 2 * channels[i-1])/VECTOR_SIZE);
            offsets[i] = cumulative;
        end
        return offsets;
    endfunction

    // Function: calc_weights_per_layer
    // Description: Computes total number of weights for each layer in the model
    //              Returns array where each index is the number of weights in that layer
    // Arguments:
    //   layer_sizes - Kernel dimensions for each layer
    //   channels    - Input channels for each layer
    function automatic calc_weights_per_layer(input int layer_sizes[], input int channels[]);
        automatic int weights_per_layer[NUM_LAYERS];
        for (int i = 0; i < NUM_LAYERS; i++) begin
            weights_per_layer[i] = layer_sizes[i] ** 2 * channels[i];
        end
        return weights_per_layer;
    endfunction

    // Starting indices for each layer in the flat ROM array
    localparam int LAYER_OFFSETS [NUM_LAYERS] = calc_layer_offsets(LAYER_MATRIX_SIZES, LAYER_CHANNELS);
    // Number of weights in each layer of the model
    localparam int WEIGHTS_PER_LAYER [NUM_LAYERS] = calc_weights_per_layer(LAYER_MATRIX_SIZES, LAYER_CHANNELS);
    // Total number of weights across all layers
    localparam int TOTAL_WEIGHTS = LAYER_OFFSETS[NUM_LAYERS-1] + 
                                  (LAYER_MATRIX_SIZES[NUM_LAYERS-1]**2 * 
                                  LAYER_CHANNELS[NUM_LAYERS-1]);
    // Parameter used for more efficient multiplication with vector size
    localparam int VECTOR_SHIFT = $clog2(VECTOR_SIZE);

    // Block RAM storing all weights in layer-major order.
    // Stores weights in 16 weight vectors
    logic [(WEIGHT_WIDTH*VECTOR_SIZE)-1:0] weight_rom [$ceil(TOTAL_WEIGHTS/VECTOR_SIZE)];
    initial begin
        foreach(weight_rom[i]) begin
            // weight_rom[i] = ...; // Initialize ROM with weight values
        end
    end

    // Combines layer offset with intra-layer address, prevent out-of-bounds access
    logic [$clog2(TOTAL_WEIGHTS/VECTOR_SIZE)-1:0] calc_addr;
    always_comb begin
        calc_addr = LAYER_OFFSETS[layer_sel] + addr;
        if (calc_addr >= $ceil(TOTAL_WEIGHTS/VECTOR_SIZE))
            calc_addr = '0;
    end

    // Read appropriate ROM values, set valid bits for out-of-layer accesses
    always_ff @(posedge clk) begin
        if (!read_valid) begin
            valid_out  <= '0;
            weight_out <= '0;
        end else begin
            // Output is valid if it is within bounds of layer_sel's weight count
            weight_out[i] <= weight_rom[calc_addr];
            for (int i = 0; i < 15; i++) begin
                valid_out[i]  <= (addr<<VECTOR_SHIFT + i < WEIGHTS_PER_LAYER[layer_sel]) ? 1'b1 : 1'b0;
            end
        end
    end

endmodule
