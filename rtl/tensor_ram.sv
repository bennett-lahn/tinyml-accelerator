
// 4 32 bit output busses. 16 8 bit pixels per read

//data is stored like such:
// row0 | row1 | row2 | row3
//and in each row:
// pixel0 | pixel1 | pixel2 | pixel3
// MSB------------------------------LSB

module tensor_ram #(
    parameter D_WIDTH = 128, //(4 8 bit pixels per read/write)
    parameter DEPTH = 96*96,
    parameter INIT_FILE = ""
) (
    input logic clk
    ,input logic we
    ,input logic [$clog2(DEPTH)-1:0] addr_w
    ,input logic [D_WIDTH-1:0] din
    ,input logic [$clog2(DEPTH)-1:0] addr_r
    ,output logic [31:0] dout0
    ,output logic [31:0] dout1
    ,output logic [31:0] dout2
    ,output logic [31:0] dout3
);

    logic [D_WIDTH-1:0] ram [0:DEPTH-1];


    initial begin
        if (INIT_FILE != "") begin // Only initialize if a file is specified (using the new parameter)
            $display("tensor_ram: Initializing RAM from file: %s", INIT_FILE);
            $readmemh(INIT_FILE, ram);
        end else begin
            //Default initialization if no file is provided, e.g., all zeros
            for (int i = 0; i < DEPTH; i++) begin
                ram[i] = {D_WIDTH{1'b0}};
            end
            $display("tensor_ram: No INIT_FILE specified, RAM not initialized from file by $readmemh.");
        end
    end

    //never writing or reading to the same memory location :)
    always_ff @(posedge clk) begin
        if (we) begin
            ram[addr_w] <= din;
        end
        else begin

            dout3 <= ram[addr_r][31:0];
            dout2 <= ram[addr_r][63:32];
            dout1 <= ram[addr_r][95:64];
            dout0 <= ram[addr_r][127:96];
        end


    end




endmodule
