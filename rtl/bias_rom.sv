// Parameterizable synchronous ROM for storing CNN layer bias values.
// Each layer has a fixed number of bias values equal to its channel count.
// Supports multiple layers with independent channel counts.

module bias_rom #(
    parameter int NUM_LAYERS                           // Number of CNN layers
    ,parameter int LAYER_CHANNELS [NUM_LAYERS]         // Number of channels per layer (bias count per layer)
    ,parameter int BIAS_WIDTH = 32                     // Bit width of each bias value
)(
    input  logic clk                                   // Clock for synchronous read
    ,input  logic [$clog2(NUM_LAYERS)-1:0] layer_sel   // Layer select: selects which layer's bias to access
    ,input  logic [$clog2(LAYER_CHANNELS[0])-1:0] addr // Address within the selected layer (channel index)

    ,output logic [BIAS_WIDTH-1:0] bias_out            // bias value at the specified layer and channel
);

    // Function: calc_layer_offsets
    // Description: Computes the starting offset in the bias ROM for each layer.
    //              Returns an array where offsets[i] is the index of the first
    //              bias value for layer i.
    function automatic int calc_layer_offsets(input int channels[]);
        automatic int offsets[NUM_LAYERS];
        automatic int cumulative = 0;
        for (int i = 0; i < NUM_LAYERS; i++) begin
            offsets[i] = cumulative;
            cumulative += channels[i];
        end
        return offsets;
      endfunction

    // Localparams: LAYER_OFFSETS, TOTAL_BIASES
    // Description: 
    //   - LAYER_OFFSETS: Array of starting indices for each layer's biases.
    //   - TOTAL_BIASES: Total number of bias values stored in the ROM.
    localparam int LAYER_OFFSETS [NUM_LAYERS] = calc_layer_offsets(LAYER_CHANNELS);
    localparam int TOTAL_BIASES = LAYER_OFFSETS[NUM_LAYERS-1] + LAYER_CHANNELS[NUM_LAYERS-1];

    // ROM Storage: bias_rom
    // Description: Synchronous ROM array storing all bias values.
    //              Initialized from BIAS_VALUES parameter.
    logic [BIAS_WIDTH-1:0] bias_rom [TOTAL_BIASES];
    initial begin
        foreach(bias_rom[i]) begin
            bias_rom[i] = BIAS_VALUES[i];
        end
    end

    // Address Calculation
    // Description: Computes the absolute address in the bias ROM for the
    //              selected layer and channel index (addr).
    //              If address is out of bounds, wraps to zero.
    logic [$clog2(TOTAL_BIASES)-1:0] calc_addr;
    always_comb begin
        calc_addr = LAYER_OFFSETS[layer_sel] + addr;
        if (calc_addr >= TOTAL_BIASES)
            calc_addr = 0;
    end

    // Synchronous ROM Read
    always_ff @(posedge clk) begin
        bias_out <= bias_rom[calc_addr];
    end

endmodule
