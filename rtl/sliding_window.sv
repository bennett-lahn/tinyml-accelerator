`include "./sys_types.svh"
// Module behavior:
// Takes in a 32-bit bus (4 pixels of type int8_t) per clock cycle when valid_in is high.
// This 32-bit bus is treated as one row of 4 pixels.
// The module buffers the last 4 such rows.
// Individual valid signals indicate when each output row is ready, with valid_A0 asserting first.
// A0 = newest of the 4 buffered rows, A1 = next newest, ..., A3 = oldest row.
//
// Example Trace: Input bus sequence (each is 4 pixels): C0, C1, C2, C3...
// - After C0 input (valid_in high):
//   A0=C0, valid_A0=1. Other A_x invalid.
// - After C1 input (valid_in high):
//   A0=C1, valid_A0=1.
//   A1=C0, valid_A1=1. Other A_x invalid.
// - After C2 input (valid_in high):
//   A0=C2, valid_A0=1.
//   A1=C1, valid_A1=1.
//   A2=C0, valid_A2=1. valid_A3 invalid.
// - After C3 input (valid_in high):
//   A0=C3, valid_A0=1. (Newest row)
//   A1=C2, valid_A1=1.
//   A2=C1, valid_A2=1.
//   A3=C0, valid_A3=1. (Oldest row of this 4-block window)
//   At this point, the 4x4 window is (A0=C3, A1=C2, A2=C1, A3=C0).

module sliding_window  
    (
    input logic clk,
    input logic reset,
    input logic valid_in,                   // Indicates validity of pixels_in_chunk_bus
    input logic [31:0] pixels_in_chunk_bus, // 4 pixels (int8_t) packed: [P3,P2,P1,P0] where P0 is bits 7:0
    output int8_t A0 [0:3],                 // Row 0 of the 4x4 window (newest of the 4 buffered rows)
    output int8_t A1 [0:3],                 // Row 1 of the 4x4 window (1-cycle old row)
    output int8_t A2 [0:3],                 // Row 2 of the 4x4 window (2-cycles old row)
    output int8_t A3 [0:3],                 // Row 3 of the 4x4 window (oldest of the 4 buffered rows)
    output logic valid_A0,                  // Indicates A0 (newest row data) is valid
    output logic valid_A1,                  // Indicates A1 (1-cycle old row data) is valid
    output logic valid_A2,                  // Indicates A2 (2-cycles old row data) is valid
    output logic valid_A3                   // Indicates A3 (3-cycles old row data) is valid
    );

    // Internal buffers to store the last 4 rows.
    // row_buf0 stores the most recently received valid row (becomes A0).
    // row_buf1 stores the row received one valid_in cycle before row_buf0 (becomes A1).
    // row_buf2 stores the row received two valid_in cycles before row_buf0 (becomes A2).
    // row_buf3 stores the row received three valid_in cycles before row_buf0 (becomes A3).
    int8_t row_buf0 [0:3]; // Newest
    int8_t row_buf1 [0:3];
    int8_t row_buf2 [0:3];
    int8_t row_buf3 [0:3]; // Oldest

    // Pipeline to delay valid_in. These will be used to generate individual valid signals.
    logic valid_in_d1, valid_in_d2, valid_in_d3;

    always_ff @(posedge clk) begin
        if (reset) begin
            row_buf0 <= '{default:'0};
            row_buf1 <= '{default:'0};
            row_buf2 <= '{default:'0};
            row_buf3 <= '{default:'0};
            valid_in_d1 <= 1'b0;
            valid_in_d2 <= 1'b0;
            valid_in_d3 <= 1'b0;
        end else begin
            // If valid_in is high, new data is loaded into row_buf0,
            // and existing data shifts down the pipeline.
            // If valid_in is low, buffers hold their values.
            if (valid_in) begin
                // Shift existing rows: row_buf0 -> row_buf1 -> row_buf2 -> row_buf3
                row_buf3 <= row_buf2;
                row_buf2 <= row_buf1;
                row_buf1 <= row_buf0;

                // Load new incoming row into row_buf0 (the newest row buffer)
                // Pixel mapping: bus LSB (bits 7:0) is pixel 0 of the row.
                // A_X[0] should correspond to pixels_in_chunk_bus[7:0] for that row.
                row_buf0[0] <= int8_t'(pixels_in_chunk_bus[7:0]);   // Pixel 0
                row_buf0[1] <= int8_t'(pixels_in_chunk_bus[15:8]);  // Pixel 1
                row_buf0[2] <= int8_t'(pixels_in_chunk_bus[23:16]); // Pixel 2
                row_buf0[3] <= int8_t'(pixels_in_chunk_bus[31:24]); // Pixel 3
            end

            // Pipeline the valid_in signal
            valid_in_d1 <= valid_in;
            valid_in_d2 <= valid_in_d1;
            valid_in_d3 <= valid_in_d2;
        end
    end

    // Assign outputs: A0 is newest, A3 is oldest
    assign A0 = row_buf0;
    assign A1 = row_buf1;
    assign A2 = row_buf2;
    assign A3 = row_buf3;

    // Assign individual valid signals for each output row
    // valid_A0 is true if current input is valid
    // valid_A1 is true if input 1 cycle ago was valid
    // valid_A2 is true if input 2 cycles ago was valid
    // valid_A3 is true if input 3 cycles ago was valid (completing the 4-stage buffer)
    assign valid_A0 = valid_in;
    assign valid_A1 = valid_in_d1;
    assign valid_A2 = valid_in_d2;
    assign valid_A3 = valid_in_d3;

endmodule
