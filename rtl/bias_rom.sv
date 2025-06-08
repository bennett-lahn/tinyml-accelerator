// Parameterizable synchronous ROM for storing CNN layer bias values.
// Each layer has a fixed number of bias values equal to its output channel count.
// Supports multiple layers with independent channel counts accessed via [layer][output channel] indexing.
// ROM is packed without gaps - addresses are calculated using cumulative channel counts.

module bias_rom #(
    parameter WIDTH                    = 32                                       // Bit width of bias values
    ,parameter DEPTH                   = 240                                      // Total number of bias values across all layers
    ,parameter NUM_LAYERS              = 6                                        // Number of CNN layers
    ,parameter MAX_NUM_CH              = 64                                       // Maximum number of output channels per layer
    ,parameter CH_BITS                 = $clog2(MAX_NUM_CH)                       // Number of bits required to address all channels
    ,parameter int CONV_OUT_C [NUM_LAYERS] = '{8, 16, 32, 64, 64, 10}            // Output channels per layer
    ,parameter INIT_FILE               = "../fakemodel/tflite_bias_weights.hex"   // Initialization file path
)(
    input  logic                          clk                                     // Clock for synchronous read
    ,input  logic                         read_enable                             // High if read request is valid
    ,input  logic [$clog2(NUM_LAYERS)-1:0]       layer_idx                       // Layer index
    ,input  logic [CH_BITS-1:0]               channel_idx                     // Output channel index within layer

    ,output logic [WIDTH-1:0]            bias_out                                // Bias value for specified layer and channel
);

    logic [WIDTH-1:0] rom [0:DEPTH-1];
    logic [$clog2(DEPTH)-1:0] linear_addr;
    logic [$clog2(DEPTH)-1:0] layer_offset_lut [NUM_LAYERS];

    // Precompute layer base addresses in a lookup table
    initial begin
        layer_offset_lut[0] = 0;  // First layer starts at address 0
        for (int i = 1; i < NUM_LAYERS; i++) begin
            layer_offset_lut[i] = layer_offset_lut[i-1] + CONV_OUT_C[i-1];
        end
    end

    // Calculate final linear address by adding channel offset to layer base
    assign linear_addr = layer_offset_lut[layer_idx] + channel_idx;

    initial begin
        if (INIT_FILE != "") begin
            $display("bias_rom: Initializing ROM from file: %s", INIT_FILE);
            $readmemh(INIT_FILE, rom);
        end else begin
            // Default initialization if no file is provided - all zeros
            for (int i = 0; i < DEPTH; i++) begin
                rom[i] = {WIDTH{1'b0}};
            end
            $display("bias_rom: No INIT_FILE specified, ROM not initialized from file by $readmemh.");
        end
    end

    // // Address bounds checking
    // always_ff @(posedge clk) begin
    //     if (read_enable && linear_addr >= DEPTH) begin
    //         $display("ERROR: bias_rom linear address out of bounds at time %0t! layer_idx=%d, channel_idx=%d, linear_addr=%d, max_valid_addr=%d", 
    //                  $time, layer_idx, channel_idx, linear_addr, DEPTH-1);
    //     end
    //     if (read_enable && layer_idx >= NUM_LAYERS) begin
    //         $display("ERROR: bias_rom layer_idx out of bounds at time %0t! layer_idx=%d, max_layer_idx=%d", 
    //                  $time, layer_idx, NUM_LAYERS-1);
    //     end
    //     if (read_enable && channel_idx >= CONV_OUT_C[layer_idx]) begin
    //         $display("ERROR: bias_rom channel_idx out of bounds at time %0t! layer_idx=%d, channel_idx=%d, max_channel_idx=%d", 
    //                  $time, layer_idx, channel_idx, CONV_OUT_C[layer_idx]-1);
    //     end
    // end

    // Synchronous read operation
    always_ff @(posedge clk) begin
        if (read_enable) begin
            bias_out <= rom[linear_addr];
        end else begin
            bias_out <= {WIDTH{1'b0}};
        end
    end
   
endmodule
