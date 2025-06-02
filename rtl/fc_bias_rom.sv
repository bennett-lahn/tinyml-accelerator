// Parameterizable synchronous ROM for storing fully connected layer bias values.
// Stores biases for the final two fully connected layers only.
// Supports multiple FC layers with independent output counts.

module fc_bias_rom #(
    parameter WIDTH = 32,            // Width of bias values (typically 32-bit for int32 biases)
    parameter DEPTH = 74,           // Total depth to accommodate both FC layers
    parameter INIT_FILE = "../fakemodel/tflite_fc_biases.hex",        // Optional initialization file for FC biases
    parameter FC1_SIZE = 64,        // Number of neurons in first FC layer
    parameter FC2_SIZE = 10          // Number of neurons in second FC layer (output layer)
)(
    input  logic clk,                                   // Clock for synchronous read
    input  logic read_enable,                           // High if read request is valid
    input  logic fc_layer_select,                       // 0: FC1, 1: FC2 (output layer)
    input  logic [$clog2(DEPTH)-1:0] addr,            // Address within the selected FC layer

    output logic [WIDTH-1:0] bias_out                  // Bias value at the specified FC layer and neuron
);

    logic [WIDTH-1:0] rom [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin // Only initialize if a file is specified
            $display("fc_bias_rom: Initializing ROM from file: %s", INIT_FILE);
            $readmemh(INIT_FILE, rom);
            // Debug: print first few values to verify loading
            $display("fc_bias_rom: First 4 values: %h %h %h %h", rom[0], rom[1], rom[2], rom[3]);
        end else begin
            // Default initialization if no file is provided, e.g., all zeros
            for (int i = 0; i < DEPTH; i++) begin
                rom[i] = {WIDTH{1'b0}};
            end
            $display("fc_bias_rom: No INIT_FILE specified, ROM not initialized from file by $readmemh.");
        end
    end

    // Address calculation for FC layer selection
    logic [$clog2(DEPTH)-1:0] actual_addr;
    always_comb begin
        if (fc_layer_select == 1'b0) begin
            // FC1 layer: addresses 0 to FC1_SIZE-1
            actual_addr = addr;
        end else begin
            // FC2 layer: addresses FC1_SIZE to FC1_SIZE+FC2_SIZE-1
            actual_addr = ($clog2(DEPTH))'(FC1_SIZE) + addr;
        end
    end

    // Address bounds checking
    always_ff @(posedge clk) begin
        if (read_enable && actual_addr >= DEPTH) begin
            $display("ERROR: fc_bias_rom address out of bounds at time %0t! actual_addr=%d, max_valid_addr=%d, fc_layer_select=%b, addr=%d", 
                     $time, actual_addr, DEPTH-1, fc_layer_select, addr);
        end
    end

    always_ff @(posedge clk) begin
        if (read_enable) begin
            bias_out <= rom[actual_addr];
        end
    end
   
endmodule 
