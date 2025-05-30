`include "./sys_types.svh"
// Assuming sys_types.svh might contain:
// typedef logic signed [7:0]   int8_t;
// typedef logic signed [15:0] int16_t;
// typedef logic signed [31:0] int32_t; // Or use logic [31:0] directly

// Module behavior:
// Takes in a 32-bit bus (4 pixels of type int8_t) per clock cycle when valid_in is high.
// Each valid input chunk is sequentially routed to A0, then A1, then A2, then A3.
// valid_A0 pulses when A0 is loaded.
// valid_A1 pulses one cycle later when A1 is loaded with the next chunk.
// ...and so on for valid_A2 and valid_A3.
// After 4 valid inputs (C0, C1, C2, C3), the outputs will hold:
// A0=C0, A1=C1, A2=C2, A3=C3.
//
// Example Trace: Input bus sequence (each is 4 pixels): C0, C1, C2, C3...
// - After C0 input (valid_in high, current_input_lane_idx was 0):
//   A0=C0, valid_A0=1. Other A_x hold previous data, other valid_Ax=0.
// - After C1 input (valid_in high, current_input_lane_idx was 1):
//   A1=C1, valid_A1=1. Other A_x hold previous data, other valid_Ax=0 (except A0 from previous).
// - After C2 input (valid_in high, current_input_lane_idx was 2):
//   A2=C2, valid_A2=1.
// - After C3 input (valid_in high, current_input_lane_idx was 3):
//   A3=C3, valid_A3=1.
//   At this point, A0=C0, A1=C1, A2=C2, A3=C3. The valid signals pulsed sequentially.

module sliding_window(
    input logic clk,
    input logic reset,
    input logic start,
    input logic valid_in,                   // Indicates validity of pixels_in_chunk_bus
    input logic [31:0] A0_in,
    input logic [31:0] A1_in,
    input logic [31:0] A2_in,
    input logic [31:0] A3_in,               // 4x8 bit pixels per input chunk
    output int8_t A0 [0:3],                 // Row 0 of the 4x4 window
    output int8_t A1 [0:3],                 // Row 1 of the 4x4 window
    output int8_t A2 [0:3],                 // Row 2 of the 4x4 window
    output int8_t A3 [0:3],                 // Row 3 of the 4x4 window
    output logic valid_A0,                  // Pulse when A0 is loaded
    output logic valid_A1,                  // Pulse when A1 is loaded
    output logic valid_A2,                  // Pulse when A2 is loaded
    output logic valid_A3,                   // Pulse when A3 is loaded
    output logic done
);

    // Registers to store each row of the 4x4 block
    int8_t A0_data [0:3];
    int8_t A1_data [0:3];
    int8_t A2_data [0:3];
    int8_t A3_data [0:3];

    // Internal valid pulse generators for each lane
    logic vA0_pulse, vA1_pulse, vA2_pulse, vA3_pulse;

    // Counter to determine which output lane (A0-A3) the current input is for
    logic [1:0] current_input_lane_idx;

    
    // Pipeline the 4 input rows so that A0_in is output first and then each subsequent row one clock cycle later.
    logic active_row;
    logic [2:0] row_sel;

    always_ff @(posedge clk) begin
        if (reset) begin
            active_row   <= 1'b0;
            row_sel      <= 3'd0;
            vA0_pulse    <= 1'b0;
            vA1_pulse    <= 1'b0;
            vA2_pulse    <= 1'b0;
            vA3_pulse    <= 1'b0;
            A0_data      <= '{default:'0};
            A1_data      <= '{default:'0};
            A2_data      <= '{default:'0};
            A3_data      <= '{default:'0};
            done         <= 1'b0;
        end else begin
            // Clear valid pulses every cycle
            vA0_pulse <= 1'b0;
            vA1_pulse <= 1'b0;
            vA2_pulse <= 1'b0;
            vA3_pulse <= 1'b0;
            // Clear outputs every cycle
            // A0_data <= '{default:'0};
            // A1_data <= '{default:'0};
            // A2_data <= '{default:'0};
            // A3_data <= '{default:'0};

            // On start, begin the pipeline
            if (start) begin
                active_row <= 1'b1;
                row_sel    <= 3'd0;
                done       <= 1'b0;
            end
            
            if (active_row) begin
                case (row_sel)
                    3'd0: begin
                        A0_data[3] <= A0_in[7:0];
                        A0_data[2] <= A0_in[15:8];
                        A0_data[1] <= A0_in[23:16];
                        A0_data[0] <= A0_in[31:24];
                        vA0_pulse  <= 1'b1;
                        row_sel    <= row_sel + 1;
                        done <= 1'b0;
                    end
                    3'd1: begin
                        A1_data[3] <= A1_in[7:0];
                        A1_data[2] <= A1_in[15:8];
                        A1_data[1] <= A1_in[23:16];
                        A1_data[0] <= A1_in[31:24];
                        vA1_pulse  <= 1'b1;
                        row_sel    <= row_sel + 1;
                        A0_data[0] <= 0;
                        A0_data[1] <= 0;
                        A0_data[2] <= 0;
                        A0_data[3] <= 0; // Clear A0 data after it has been used
                        done <= 1'b0;
                    end
                    3'd2: begin
                        A2_data[3] <= A2_in[7:0];
                        A2_data[2] <= A2_in[15:8];
                        A2_data[1] <= A2_in[23:16];
                        A2_data[0] <= A2_in[31:24];
                        vA2_pulse  <= 1'b1;
                        row_sel    <= row_sel + 1;
                        A1_data[0] <= 0;
                        A1_data[1] <= 0;
                        A1_data[2] <= 0;
                        A1_data[3] <= 0; // Clear A1 data after it has been used
                        done <= 1'b0;
                    end
                    3'd3: begin
                        A3_data[3] <= A3_in[7:0];
                        A3_data[2] <= A3_in[15:8];
                        A3_data[1] <= A3_in[23:16];
                        A3_data[0] <= A3_in[31:24];
                        vA3_pulse  <= 1'b1;
                        A2_data[0] <= 0;
                        A2_data[1] <= 0;
                        A2_data[2] <= 0;
                        A2_data[3] <= 0; 
                        row_sel   <= row_sel + 1; // Move to the next row
                        // Clear A2 data after it has been used
                        // active_row <= 1'b0; // End pipeline after the 4th row
                        done <= 1'b0;
                    end
                    3'd4: begin
                        A3_data[0] <= 0;
                        A3_data[1] <= 0;
                        A3_data[2] <= 0;
                        A3_data[3] <= 0; // Clear A3 data after it has been used
                        active_row <= 1'b0; // End pipeline after the 4th row
                        done <= 1'b1; // Indicate that the sliding window operation is done
                    end
                    default: begin
                        // Do nothing, just keep the last row data
                        active_row <= 1'b0; // End pipeline after the 4th row
                    end


                endcase
            end
        end
    end

    // Assign outputs
    assign A0 = A0_data;
    assign A1 = A1_data;
    assign A2 = A2_data;
    assign A3 = A3_data;

    assign valid_A0 = vA0_pulse;
    assign valid_A1 = vA1_pulse;
    assign valid_A2 = vA2_pulse;
    assign valid_A3 = vA3_pulse;

endmodule