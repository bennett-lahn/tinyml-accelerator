module tensor_ram #(
    parameter D_WIDTH = 8,
    parameter DEPTH = 96*96
) (
    input logic clk
    ,input logic we
    ,input logic [$clog2(DEPTH)-1:0] addr_w
    ,input logic [D_WIDTH-1:0] din
    ,input logic [$clog2(DEPTH)-1:0] addr_r
    ,output logic [D_WIDTH-1:0] dout
);

    logic [D_WIDTH-1:0] ram [0:DEPTH-1];

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