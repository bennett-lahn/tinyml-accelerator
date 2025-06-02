module dense_weight_rom 
#(
  parameter DEPTH = 16384,
  parameter WIDTH = 32,
  parameter INIT_FILE = "dense_weights.hex"
)
(
  input logic clk
  ,input logic reset
  ,input logic [$clog2(DEPTH)-1:0] addr
  ,input logic read_enable
  ,output logic [WIDTH-1:0] weight_out
);


  // Initialize the ROM with weights
  initial begin
    if (INIT_FILE != "") begin
      $display("dense_weight_rom: Initializing ROM from file: %s", INIT_FILE);
      $readmemh(INIT_FILE, rom);
    end else begin
      $display("dense_weight_rom: No INIT_FILE specified, initializing ROM to zero");
      for (int i = 0; i < DEPTH; i++) begin
        rom[i] = {WIDTH{1'b0}};
      end
    end
  end

    logic [WIDTH-1:0] rom [DEPTH];

    always_ff @(posedge clk) begin
        if (read_enable) begin
            weight_out <= rom[addr];
        end
    end
endmodule



