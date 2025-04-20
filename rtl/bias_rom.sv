// Module: bias_rom
// Description: Parameterizable synchronous ROM for storing CNN layer bias values.
//              Each layer has a fixed number of bias values equal to its channel count.
//              Supports multiple layers with independent channel counts.

// TODO: Should ROMs be expected to return data in one cycle?
module bias_rom #(
    // Number of CNN layers
    parameter int NUM_LAYERS,

    // Array: Number of channels per layer (bias count per layer)
    parameter int LAYER_CHANNELS [NUM_LAYERS-1:0],

    // Bit width of each bias value
    parameter int BIAS_WIDTH = 32,

    // Packed array of all bias values for all layers, in layer-major order.
    // Size must equal sum of all LAYER_CHANNELS.
    parameter logic [BIAS_WIDTH-1:0] BIAS_VALUES []
)(
    // Clock input for synchronous read
    input  logic clk,

    // Layer select: selects which layer's bias to access
    input  logic [$clog2(NUM_LAYERS)-1:0] layer_sel,

    // Address within the selected layer (channel index)
    input  logic [$clog2(LAYER_CHANNELS[0])-1:0] addr,

    // Output: bias value at the specified layer and channel
     output logic [BIAS_WIDTH-1:0] bias_out
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
