module tensor_ram #(
    parameter D_WIDTH = 32, //(4 8 bit pixels per read/write)
    parameter DEPTH = 96*96,
    parameter INIT_FILE = ""
) (
    input logic clk
    ,input logic we
    ,input logic [$clog2(DEPTH)-1:0] addr_w
    ,input logic [D_WIDTH-1:0] din
    ,input logic [$clog2(DEPTH)-1:0] addr_r
    ,output logic [D_WIDTH-1:0] dout
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

    always_ff @(posedge clk) begin
        if (we) begin
            ram[addr_w] <= din;
        end
        if(we && (addr_w == addr_r)) begin
            dout <= din;
        end
        else
            dout <= ram[addr_r];
    end




endmodule
