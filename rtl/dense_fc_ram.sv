module dense_fc_ram
#(
  parameter DEPTH = 256
  ,parameter WIDTH = 8
  ,parameter INIT_FILE = "dense_fc_ram.hex"
)
(
  input logic clk
  ,input logic reset
  ,input logic write_enable
  ,input logic [$clog2(DEPTH)-1:0] read_addr
  ,input logic [$clog2(DEPTH)-1:0] write_addr
  ,input logic read_enable
  ,input logic [WIDTH-1:0] data_in
  ,output logic [WIDTH-1:0] data_out
);

  logic [WIDTH-1:0] ram [DEPTH];

  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, ram);
    end else begin
      for (int i = 0; i < DEPTH; i++) begin
        ram[i] = {WIDTH{1'b0}};
      end
    end
  end

  // Address bounds checking
  always_ff @(posedge clk) begin
    if ((write_enable || read_enable) && addr >= DEPTH) begin
      $display("ERROR: dense_fc_ram address out of bounds at time %0t! addr=%d, max_valid_addr=%d, write_enable=%b, read_enable=%b", 
               $time, addr, DEPTH-1, write_enable, read_enable);
    end
  end

  always_ff @(posedge clk) begin
    if (write_enable) begin
      ram[addr] <= data_in;
    end
    if (read_enable) begin
      data_out <= ram[addr];
    end
  end

endmodule 
