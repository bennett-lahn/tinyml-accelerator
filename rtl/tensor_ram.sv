`include "sys_types.svh"

module tensor_ram #(
    // Width for wide reads (bits)
    parameter int READ_WIDTH = 128,
    // Depth in number of READ_WIDTH‐bit words
    parameter int DEPTH_WORDS = 1024,
    // Width for narrow writes (bits)
    parameter int WRITE_WIDTH = 8,
    // Initialization file for simulation (hex values per READ_WIDTH word)
    parameter string INIT_FILE = ""
) (
    input  logic                        clk,
    input  logic                        we,        // write enable
    input  logic                        re,        // read enable
    // Write address: byte‐granular over entire array
    input  logic [$clog2(DEPTH_WORDS*(READ_WIDTH/WRITE_WIDTH))-1:0] addr_w,
    input  logic [WRITE_WIDTH-1:0]      din,
    // Read address: word‐granular
    input  logic [$clog2(DEPTH_WORDS)-1:0]             addr_r,
    output logic [READ_WIDTH-1:0]       dout,
    
    // Additional outputs for systolic array buffering
    // Each 32-bit output contains 4 consecutive 8-bit pixels
    output logic [31:0]                 dout0,     // bytes [3:0]
    output logic [31:0]                 dout1,     // bytes [7:4] 
    output logic [31:0]                 dout2,     // bytes [11:8]
    output logic [31:0]                 dout3      // bytes [15:12]
);

    // Number of bytes per READ_WIDTH word
    localparam int BYTES_PER_WORD = READ_WIDTH / WRITE_WIDTH;

    // Underlying memory: each entry is a READ_WIDTH‐bit word
    logic [READ_WIDTH-1:0] ram [0:DEPTH_WORDS-1];

    // Optional initialization for simulation
    initial begin
        if (INIT_FILE != "") begin
            $display("tensor_ram: initializing from %s", INIT_FILE);
            $readmemh(INIT_FILE, ram);
        end else begin
            for (int i = 0; i < DEPTH_WORDS; i++) begin
                ram[i] = '0;
            end
        end
    end

    // Synchronous read/write
    always_ff @(posedge clk) begin
        // Write one byte into the wide word
        if (we) begin
            // Derive which word and which byte lane
            /* verilator lint_off WIDTHEXPAND */
            /* verilator lint_off WIDTHTRUNC */
            int word_idx = {18'd0, addr_w} / BYTES_PER_WORD;
            int byte_idx = {18'd0, addr_w} % BYTES_PER_WORD;
            /* verilator lint_on WIDTHTRUNC */
            /* verilator lint_on WIDTHEXPAND */
            // Write-enable only that byte slice
            ram[word_idx][ byte_idx*WRITE_WIDTH +: WRITE_WIDTH ] <= din;
        end
        // Read entire wide word
        if (re) begin
            dout <= ram[addr_r];
            // Break down 128-bit word into 4x 32-bit chunks for systolic array
            // Each 32-bit chunk contains 4 consecutive 8-bit pixels
            dout0 <= ram[addr_r][31:0];     // bytes [3:0]
            dout1 <= ram[addr_r][63:32];    // bytes [7:4]
            dout2 <= ram[addr_r][95:64];    // bytes [11:8] 
            dout3 <= ram[addr_r][127:96];   // bytes [15:12]
        end
    end

endmodule
