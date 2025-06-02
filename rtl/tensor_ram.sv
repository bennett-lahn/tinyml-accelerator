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
    ,output logic [127:0]                             data_out
    
    // 32-bit outputs: data3 is LSB, data0 is MSB of 128-bit word
    ,output logic [31:0]                              ram_dout0      // MSB [127:96]
    ,output logic [31:0]                              ram_dout1      // [95:64]
    ,output logic [31:0]                              ram_dout2      // [63:32]
    ,output logic [31:0]                              ram_dout3      // LSB [31:0]
);

    localparam int NUM_BANKS = 16;
    localparam int BANK_WIDTH_BITS = 8;
    localparam int DOUT_WIDTH_BITS = NUM_BANKS * BANK_WIDTH_BITS; // Should be 128

    // Total number of 8-bit entries in each bank
    // This is equivalent to the number of 128-bit words in the whole RAM
    localparam int BANK_DEPTH_ENTRIES = DEPTH_128B_WORDS;

    // Memory banks: NUM_BANKS banks, each BANK_DEPTH_ENTRIES deep, BANK_WIDTH_BITS wide
    logic [BANK_WIDTH_BITS-1:0] banks [NUM_BANKS-1:0] [BANK_DEPTH_ENTRIES-1:0];

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
    end

    // Calculate write address using row-major, channel-last order
    // write_addr = (row * num_cols * num_channels) + (col * num_channels) + channel
    logic [$clog2(DEPTH_128B_WORDS * 16)-1:0] write_addr;
    always_comb begin
        write_addr = (ADDR_BITS'(write_row) * ADDR_BITS'(num_cols) * ADDR_BITS'(num_channels)) + 
                    (ADDR_BITS'(write_col) * ADDR_BITS'(num_channels)) + 
                    ADDR_BITS'(write_channel);
    end

    // Address bounds checking
    always_ff @(posedge clk) begin
        // Check read address bounds
        if (read_en && read_addr >= DEPTH_128B_WORDS) begin
            $display("ERROR: tensor_ram read address out of bounds at time %0t! read_addr=%d, max_valid_read_addr=%d", 
                     $time, read_addr, DEPTH_128B_WORDS-1);
        end
        
        // Check write address bounds - write_addr addresses individual bytes, so max is DEPTH_128B_WORDS * 16 - 1
        if (write_en && write_addr >= (DEPTH_128B_WORDS * 16)) begin
            $display("ERROR: tensor_ram write address out of bounds at time %0t! write_addr=%d, max_valid_write_addr=%d, row=%d, col=%d, channel=%d", 
                     $time, write_addr, (DEPTH_128B_WORDS * 16) - 1, write_row, write_col, write_channel);
        end
    end

    // The lower $clog2(NUM_BANKS) bits of write_addr select the bank.
    // The upper bits of write_addr select the address (word) within that bank.
    logic [$clog2(NUM_BANKS)-1:0]          bank_select_for_write;
    logic [$clog2(BANK_DEPTH_ENTRIES)-1:0] bank_address_for_write;

    always_ff @(posedge clk) begin
        // Write one byte into the appropriate bank and location
        if (write_en) begin
            bank_select_for_write  = write_addr[$clog2(NUM_BANKS)-1:0];
            bank_address_for_write = write_addr >> $clog2(NUM_BANKS);
            
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
            
            // Split 128-bit output into 4 32-bit words
            // data3 is LSB, data0 is MSB
            ram_dout3 <= data_out[31:0];     // LSB [31:0]
            ram_dout2 <= data_out[63:32];    // [63:32]
            ram_dout1 <= data_out[95:64];    // [95:64]
            ram_dout0 <= data_out[127:96];   // MSB [127:96]
        end
    end

endmodule
