`include "sys_types.svh"

module tensor_ram #(
    // Depth in number of 128-bit words
    parameter int DEPTH_128B_WORDS = 128
    ,parameter int ADDR_BITS       = $clog2(DEPTH_128B_WORDS)
    ,parameter int MAX_N           = 64                  // Max matrix dimension for output_coordinator
    ,parameter int N_BITS          = $clog2(MAX_N)
    ,parameter int MAX_NUM_CH      = 64
    ,parameter int CH_BITS         = $clog2(MAX_NUM_CH+1)
    // Initialization file for simulation (hex values per 128-bit word)
    ,parameter string INIT_FILE    = ""
) (
    input  logic                                      clk
    ,input  logic                                     reset           // reset signal
    ,input  logic                                     write_en        // write enable
    ,input  logic                                     read_en         // read enable

    // Write addressing inputs for row-major, channel-last order
    ,input  logic [N_BITS-1:0]     write_row       // current row
    ,input  logic [N_BITS-1:0]     write_col       // current column  
    ,input  logic [CH_BITS-1:0]    write_channel   // current output channel
    ,input  logic [N_BITS-1:0]     num_cols        // number of columns per row
    ,input  logic [CH_BITS-1:0]    num_channels    // number of output channels
    ,input  int8_t                 data_in

    // Read address: 128-bit word-granular
    ,input  logic [ADDR_BITS-1:0]      read_addr
    
    // 32-bit outputs: data3 is LSB, data0 is MSB of 128-bit word
    ,output logic [31:0]                              ram_dout0      // MSB [127:96]
    ,output logic [31:0]                              ram_dout1      // [95:64]
    ,output logic [31:0]                              ram_dout2      // [63:32]
    ,output logic [31:0]                              ram_dout3      // LSB [31:0]
    ,output logic                                     data_valid     // High when ram_dout* contain valid data
);

    localparam int NUM_BANKS = 16;
    localparam int BANK_WIDTH_BITS = 8;
    localparam int DOUT_WIDTH_BITS = NUM_BANKS * BANK_WIDTH_BITS; // Should be 128

    // Total number of 8-bit entries in each bank
    // This is equivalent to the number of 128-bit words in the whole RAM
    localparam int BANK_DEPTH_ENTRIES = DEPTH_128B_WORDS;

    // Memory banks: NUM_BANKS banks, each BANK_DEPTH_ENTRIES deep, BANK_WIDTH_BITS wide
    logic [BANK_WIDTH_BITS-1:0] banks [NUM_BANKS-1:0] [BANK_DEPTH_ENTRIES-1:0];
    logic [127:0] data_out;
    
    // Debug signals for tensor_ram memory analysis
    logic [7:0] byte_values [15:0];
    logic [31:0] start_pixel_idx;
    logic [31:0] pixel_idx;
    logic [5:0] pixel_row;
    logic [5:0] pixel_col;

    // Optional initialization for simulation
    initial begin
        if (INIT_FILE != "") begin
            // Temporary memory to hold 128-bit words from file for initialization
            logic [DOUT_WIDTH_BITS-1:0] temp_mem_init [0:BANK_DEPTH_ENTRIES-1];
            $display("tensor_ram: initializing from %s", INIT_FILE);
            $readmemh(INIT_FILE, temp_mem_init);

            for (int word_idx = 0; word_idx < BANK_DEPTH_ENTRIES; word_idx++) begin
                for (int bank_idx = 0; bank_idx < NUM_BANKS; bank_idx++) begin
                    // Distribute bytes from the 128-bit word to respective banks
                    // bank_idx 0 gets the LSB byte of temp_mem_init[word_idx], bank_idx 1 gets the next byte, etc.
                    banks[bank_idx][word_idx] = temp_mem_init[word_idx][(bank_idx * BANK_WIDTH_BITS) +: BANK_WIDTH_BITS];
                end
            end
        end else begin
            for (int word_idx = 0; word_idx < BANK_DEPTH_ENTRIES; word_idx++) begin
                for (int bank_idx = 0; bank_idx < NUM_BANKS; bank_idx++) begin
                    banks[bank_idx][word_idx] = '0; // Initialize all bank entries to zero
                end
            end
        end
        
        // Display memory contents at addresses 0-7 after initialization
        $display("=== TENSOR_RAM MEMORY CONTENTS (INITIALIZATION) ===");
        $display("Address 0: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][0], banks[14][0], banks[13][0], banks[12][0], banks[11][0], banks[10][0], banks[9][0], banks[8][0], banks[7][0], banks[6][0], banks[5][0], banks[4][0], banks[3][0], banks[2][0], banks[1][0], banks[0][0]);
        $display("Address 1: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][1], banks[14][1], banks[13][1], banks[12][1], banks[11][1], banks[10][1], banks[9][1], banks[8][1], banks[7][1], banks[6][1], banks[5][1], banks[4][1], banks[3][1], banks[2][1], banks[1][1], banks[0][1]);
        $display("Address 2: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][2], banks[14][2], banks[13][2], banks[12][2], banks[11][2], banks[10][2], banks[9][2], banks[8][2], banks[7][2], banks[6][2], banks[5][2], banks[4][2], banks[3][2], banks[2][2], banks[1][2], banks[0][2]);
        $display("Address 3: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][3], banks[14][3], banks[13][3], banks[12][3], banks[11][3], banks[10][3], banks[9][3], banks[8][3], banks[7][3], banks[6][3], banks[5][3], banks[4][3], banks[3][3], banks[2][3], banks[1][3], banks[0][3]);
        $display("Address 4: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][4], banks[14][4], banks[13][4], banks[12][4], banks[11][4], banks[10][4], banks[9][4], banks[8][4], banks[7][4], banks[6][4], banks[5][4], banks[4][4], banks[3][4], banks[2][4], banks[1][4], banks[0][4]);
        $display("Address 5: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][5], banks[14][5], banks[13][5], banks[12][5], banks[11][5], banks[10][5], banks[9][5], banks[8][5], banks[7][5], banks[6][5], banks[5][5], banks[4][5], banks[3][5], banks[2][5], banks[1][5], banks[0][5]);
        $display("Address 6: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][6], banks[14][6], banks[13][6], banks[12][6], banks[11][6], banks[10][6], banks[9][6], banks[8][6], banks[7][6], banks[6][6], banks[5][6], banks[4][6], banks[3][6], banks[2][6], banks[1][6], banks[0][6]);
        $display("Address 7: 0x%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x", banks[15][7], banks[14][7], banks[13][7], banks[12][7], banks[11][7], banks[10][7], banks[9][7], banks[8][7], banks[7][7], banks[6][7], banks[5][7], banks[4][7], banks[3][7], banks[2][7], banks[1][7], banks[0][7]);
        $display("====================================================");
    end

    // Calculate write address using row-major, channel-last order
    // write_addr = (row * num_cols * num_channels) + (col * num_channels) + channel
    logic [$clog2(DEPTH_128B_WORDS * 16)-1:0] write_addr;
    always_comb begin
        /* verilator lint_off WIDTHEXPAND */
        write_addr = (ADDR_BITS'(write_row) * ADDR_BITS'(num_cols) * ADDR_BITS'(num_channels)) + 
                    (ADDR_BITS'(write_col) * ADDR_BITS'(num_channels)) + 
                    ADDR_BITS'(write_channel);
        /* verilator lint_on WIDTHEXPAND */
    end

    // Address bounds checking
    always_ff @(posedge clk) begin
        // Check read address bounds
        /* verilator lint_off WIDTHEXPAND */
        if (read_en && read_addr >= DEPTH_128B_WORDS) begin
        /* verilator lint_on WIDTHEXPAND */
            $display("ERROR: tensor_ram read address out of bounds at time %0t! read_addr=%d, max_valid_read_addr=%d", 
                     $time, read_addr, DEPTH_128B_WORDS-1);
        end
        
        // Check write address bounds - write_addr addresses individual bytes, so max is DEPTH_128B_WORDS * 16 - 1
        /* verilator lint_off WIDTHEXPAND */
        if (write_en && write_addr >= (DEPTH_128B_WORDS * 16)) begin
        /* verilator lint_on WIDTHEXPAND */
            $display("ERROR: tensor_ram write address out of bounds at time %0t! write_addr=%d, max_valid_write_addr=%d, row=%d, col=%d, channel=%d", 
                     $time, write_addr, (DEPTH_128B_WORDS * 16) - 1, write_row, write_col, write_channel);
        end
    end

    // The lower $clog2(NUM_BANKS) bits of write_addr select the bank.
    // The upper bits of write_addr select the address (word) within that bank.
    logic [$clog2(NUM_BANKS)-1:0]          bank_select_for_write;
    logic [$clog2(BANK_DEPTH_ENTRIES)-1:0] bank_address_for_write;

    assign bank_select_for_write  = write_addr[$clog2(NUM_BANKS)-1:0];
    /* verilator lint_off WIDTHTRUNC */
    assign bank_address_for_write = write_addr >> $clog2(NUM_BANKS);
    /* verilator lint_on WIDTHTRUNC */

    always_ff @(posedge clk) begin
        // Write one byte into the appropriate bank and location
        if (write_en) begin
            banks[bank_select_for_write][bank_address_for_write] <= data_in;
        end

        // Read entire 128-bit word by concatenating data from all banks at read_addr
        // read_addr directly addresses the "word line" across all banks
        if (read_en) begin
            for (int i = 0; i < NUM_BANKS; i = i + 1) begin
                // data_out bits for bank i are [(i*BANK_WIDTH_BITS) +: BANK_WIDTH_BITS]
                // e.g., bank 0 maps to data_out[7:0], bank 1 to data_out[15:8], ..., bank 15 to data_out[127:120]
                data_out[(i * BANK_WIDTH_BITS) +: BANK_WIDTH_BITS] <= banks[i][read_addr];
            end
        end
    end
    
    // Split 128-bit data_out into 4 32-bit output words
    // data3 is LSB, data0 is MSB
    assign ram_dout3 = data_out[31:0];     // LSB [31:0]
    assign ram_dout2 = data_out[63:32];    // [63:32]
    assign ram_dout1 = data_out[95:64];    // [95:64]
    assign ram_dout0 = data_out[127:96];   // MSB [127:96]
    
    // data_valid is delayed by one cycle after read_en
    always_ff @(posedge clk) begin
        if (reset) begin
            data_valid <= 1'b0;
        end else begin
            data_valid <= read_en;
        end
    end

endmodule
