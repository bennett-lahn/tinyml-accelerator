// Parameterizable synchronous ROM for storing CNN layer bias values.
// Each layer has a fixed number of bias values equal to its channel count.
// Supports multiple layers with independent channel counts.

// TODO: Intiailize ROM

module bias_rom #(
    parameter WIDTH = 32             // Number of CNN layers
    ,parameter DEPTH = 240    
    ,parameter INIT_FILE = "../fakemodel/tflite_conv_biases.hex" // Optional initialization file
)(
    input  logic clk                                   // Clock for synchronous read
    ,input  logic                          read_enable  // High if read request is valid
    ,input  logic [$clog2(DEPTH)-1:0] addr // Address within the selected layer (channel index)

    ,output logic [WIDTH-1:0] bias_out            // bias value at the specified layer and channel
);


    logic [WIDTH-1:0] rom [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin // Only initialize if a file is specified (using the new parameter)
            $display("tensor_ram: Initializing RAM from file: %s", INIT_FILE);
            $readmemh(INIT_FILE, rom);
        end else begin
            //Default initialization if no file is provided, e.g., all zeros
            for (int i = 0; i < DEPTH; i++) begin
                rom[i] = {WIDTH{1'b0}};
            end
            $display("tensor_ram: No INIT_FILE specified, RAM not initialized from file by $readmemh.");
        end
    end

    always_ff @(posedge clk) begin
        if (read_enable) begin
            bias_out <= rom[addr];
        end
        else begin
            bias_out <= 0;
        end
    end
   
endmodule
