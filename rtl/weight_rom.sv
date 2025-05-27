// Parameterizable synchronous ROM for CNN layer weights
// Stores weights for multiple layers with configurable kernel sizes and channel counts
// Returns 16 weights at a time to support STA capacity


//collum major order
/*
    for output_layer in outputs:
        for input_layer in inputs:
            load collums of weights = rom[output_layer][input_layer]
            
*/





module weight_rom #(
    parameter WIDTH = 128,
    parameter DEPTH = 16384,
    parameter INIT_FILE = "" // New parameter for initialization file
)
(
    input logic clk,
    input logic read_enable,
    input logic [$clog2(DEPTH)-1:0] addr,

    // 4 32 bit output busses
    output logic [31:0] data0,
    output logic [31:0] data1,
    output logic [31:0] data2,
    output logic [31:0] data3
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
            data3 <= rom[addr][31:0];
            data2 <= rom[addr][63:32];
            data1 <= rom[addr][95:64];
            data0 <= rom[addr][127:96];
        end
        else begin
            data0 <= 0;
            data1 <= 0;
            data2 <= 0;
            data3 <= 0;
        end
    end






endmodule

