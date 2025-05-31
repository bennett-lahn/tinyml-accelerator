`include "sys_types.svh"

module spatial_data_formatter (
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_formatting,
    input logic patches_valid,
    
    // Input: All 7x7 positions from unified buffer
    input logic [31:0] patch_pe00_in, patch_pe01_in, patch_pe02_in, patch_pe03_in, patch_pe04_in, patch_pe05_in, patch_pe06_in,
    input logic [31:0] patch_pe10_in, patch_pe11_in, patch_pe12_in, patch_pe13_in, patch_pe14_in, patch_pe15_in, patch_pe16_in,
    input logic [31:0] patch_pe20_in, patch_pe21_in, patch_pe22_in, patch_pe23_in, patch_pe24_in, patch_pe25_in, patch_pe26_in,
    input logic [31:0] patch_pe30_in, patch_pe31_in, patch_pe32_in, patch_pe33_in, patch_pe34_in, patch_pe35_in, patch_pe36_in,
    input logic [31:0] patch_pe40_in, patch_pe41_in, patch_pe42_in, patch_pe43_in, patch_pe44_in, patch_pe45_in, patch_pe46_in,
    input logic [31:0] patch_pe50_in, patch_pe51_in, patch_pe52_in, patch_pe53_in, patch_pe54_in, patch_pe55_in, patch_pe56_in,
    input logic [31:0] patch_pe60_in, patch_pe61_in, patch_pe62_in, patch_pe63_in, patch_pe64_in, patch_pe65_in, patch_pe66_in,
    
    // Output: Formatted for systolic array (4 rows streaming horizontally)
    output int8_t formatted_A0 [0:3], // Row 0, 4 channels of current column position
    output int8_t formatted_A1 [0:3], // Row 1, 4 channels of current column position  
    output int8_t formatted_A2 [0:3], // Row 2, 4 channels of current column position
    output int8_t formatted_A3 [0:3], // Row 3, 4 channels of current column position
    output logic formatted_data_valid,
    output logic all_cols_sent,
    output logic next_block
);

    // Row and column counters for complete streaming
    logic [2:0] col_counter;      // 0-6 for 7 columns
    logic [1:0] row_group_counter; // 0-3 for 4 row groups
    logic [4:0] cycle_counter;    // Overall cycle counter for staggered timing
    
    // Intermediate signals for staggered timing (to prevent latch warnings)
    logic [2:0] a1_col, a2_col, a3_col;
    logic [1:0] a1_row_group, a2_row_group, a3_row_group;
    
    // State machine for complete spatial streaming
    typedef enum logic [1:0] {
        IDLE,
        STREAMING_ROWS
    } state_t;
    state_t state;
    
    // Complete spatial streaming state machine - auto-advance every clock cycle
    always_ff @(posedge clk) begin
        if (reset) begin
            col_counter <= 0;
            row_group_counter <= 0;
            cycle_counter <= 0;
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    // Only start when both start_formatting is asserted AND patches are valid
                    // This ensures the unified buffer is fully loaded before we begin
                    if (start_formatting && patches_valid) begin
                        col_counter <= 0;
                        row_group_counter <= 0;
                        cycle_counter <= 0;
                        state <= STREAMING_ROWS;
                    end
                end
                
                STREAMING_ROWS: begin
                    // Continue only if patches remain valid (buffer data is stable)
                    if (patches_valid) begin
                        // Auto-advance every clock cycle
                        cycle_counter <= cycle_counter + 1;
                        
                        if (col_counter == 6) begin
                            col_counter <= 0;
                            if (row_group_counter == 3) begin
                                // All 4 row groups completed (28 cycles total + 3 extra for staggering)
                                if (cycle_counter >= 30) begin // 28 + 3 stagger cycles
                                    row_group_counter <= 0;
                                    cycle_counter <= 0;
                                    state <= IDLE;
                                end
                            end else begin
                                // Move to next row group
                                row_group_counter <= row_group_counter + 1;
                            end
                        end else begin
                            col_counter <= col_counter + 1;
                        end
                    end else begin
                        // If patches become invalid, pause streaming
                        // This provides robustness against timing issues
                        // Counter values are maintained so we can resume
                    end
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
    
    // Staggered data output with proper timing
    // A0 starts at cycle 0, A1 at cycle 1, A2 at cycle 2, A3 at cycle 3
    always_comb begin
        // Default assignments
        formatted_A0 = '{default: 8'h0};
        formatted_A1 = '{default: 8'h0};
        formatted_A2 = '{default: 8'h0};
        formatted_A3 = '{default: 8'h0};
        
        // Default intermediate signal assignments
        a1_col = 0;
        a1_row_group = 0;
        a2_col = 0;
        a2_row_group = 0;
        a3_col = 0;
        a3_row_group = 0;
        
        // A0 output (starts immediately at cycle 0)
        if (state == STREAMING_ROWS) begin
            case (row_group_counter)
                2'd0: begin // Row Group 0: rows 0-3
                    case (col_counter)
                        3'd0: formatted_A0 = '{patch_pe00_in[7:0], patch_pe00_in[15:8], patch_pe00_in[23:16], patch_pe00_in[31:24]};
                        3'd1: formatted_A0 = '{patch_pe01_in[7:0], patch_pe01_in[15:8], patch_pe01_in[23:16], patch_pe01_in[31:24]};
                        3'd2: formatted_A0 = '{patch_pe02_in[7:0], patch_pe02_in[15:8], patch_pe02_in[23:16], patch_pe02_in[31:24]};
                        3'd3: formatted_A0 = '{patch_pe03_in[7:0], patch_pe03_in[15:8], patch_pe03_in[23:16], patch_pe03_in[31:24]};
                        3'd4: formatted_A0 = '{patch_pe04_in[7:0], patch_pe04_in[15:8], patch_pe04_in[23:16], patch_pe04_in[31:24]};
                        3'd5: formatted_A0 = '{patch_pe05_in[7:0], patch_pe05_in[15:8], patch_pe05_in[23:16], patch_pe05_in[31:24]};
                        3'd6: formatted_A0 = '{patch_pe06_in[7:0], patch_pe06_in[15:8], patch_pe06_in[23:16], patch_pe06_in[31:24]};
                        default: formatted_A0 = '{default: 8'h0};
                    endcase
                end
                2'd1: begin // Row Group 1: rows 1-4
                    case (col_counter)
                        3'd0: formatted_A0 = '{patch_pe10_in[7:0], patch_pe10_in[15:8], patch_pe10_in[23:16], patch_pe10_in[31:24]};
                        3'd1: formatted_A0 = '{patch_pe11_in[7:0], patch_pe11_in[15:8], patch_pe11_in[23:16], patch_pe11_in[31:24]};
                        3'd2: formatted_A0 = '{patch_pe12_in[7:0], patch_pe12_in[15:8], patch_pe12_in[23:16], patch_pe12_in[31:24]};
                        3'd3: formatted_A0 = '{patch_pe13_in[7:0], patch_pe13_in[15:8], patch_pe13_in[23:16], patch_pe13_in[31:24]};
                        3'd4: formatted_A0 = '{patch_pe14_in[7:0], patch_pe14_in[15:8], patch_pe14_in[23:16], patch_pe14_in[31:24]};
                        3'd5: formatted_A0 = '{patch_pe15_in[7:0], patch_pe15_in[15:8], patch_pe15_in[23:16], patch_pe15_in[31:24]};
                        3'd6: formatted_A0 = '{patch_pe16_in[7:0], patch_pe16_in[15:8], patch_pe16_in[23:16], patch_pe16_in[31:24]};
                        default: formatted_A0 = '{default: 8'h0};
                    endcase
                end
                2'd2: begin // Row Group 2: rows 2-5
                    case (col_counter)
                        3'd0: formatted_A0 = '{patch_pe20_in[7:0], patch_pe20_in[15:8], patch_pe20_in[23:16], patch_pe20_in[31:24]};
                        3'd1: formatted_A0 = '{patch_pe21_in[7:0], patch_pe21_in[15:8], patch_pe21_in[23:16], patch_pe21_in[31:24]};
                        3'd2: formatted_A0 = '{patch_pe22_in[7:0], patch_pe22_in[15:8], patch_pe22_in[23:16], patch_pe22_in[31:24]};
                        3'd3: formatted_A0 = '{patch_pe23_in[7:0], patch_pe23_in[15:8], patch_pe23_in[23:16], patch_pe23_in[31:24]};
                        3'd4: formatted_A0 = '{patch_pe24_in[7:0], patch_pe24_in[15:8], patch_pe24_in[23:16], patch_pe24_in[31:24]};
                        3'd5: formatted_A0 = '{patch_pe25_in[7:0], patch_pe25_in[15:8], patch_pe25_in[23:16], patch_pe25_in[31:24]};
                        3'd6: formatted_A0 = '{patch_pe26_in[7:0], patch_pe26_in[15:8], patch_pe26_in[23:16], patch_pe26_in[31:24]};
                        default: formatted_A0 = '{default: 8'h0};
                    endcase
                end
                2'd3: begin // Row Group 3: rows 3-6
                    case (col_counter)
                        3'd0: formatted_A0 = '{patch_pe30_in[7:0], patch_pe30_in[15:8], patch_pe30_in[23:16], patch_pe30_in[31:24]};
                        3'd1: formatted_A0 = '{patch_pe31_in[7:0], patch_pe31_in[15:8], patch_pe31_in[23:16], patch_pe31_in[31:24]};
                        3'd2: formatted_A0 = '{patch_pe32_in[7:0], patch_pe32_in[15:8], patch_pe32_in[23:16], patch_pe32_in[31:24]};
                        3'd3: formatted_A0 = '{patch_pe33_in[7:0], patch_pe33_in[15:8], patch_pe33_in[23:16], patch_pe33_in[31:24]};
                        3'd4: formatted_A0 = '{patch_pe34_in[7:0], patch_pe34_in[15:8], patch_pe34_in[23:16], patch_pe34_in[31:24]};
                        3'd5: formatted_A0 = '{patch_pe35_in[7:0], patch_pe35_in[15:8], patch_pe35_in[23:16], patch_pe35_in[31:24]};
                        3'd6: formatted_A0 = '{patch_pe36_in[7:0], patch_pe36_in[15:8], patch_pe36_in[23:16], patch_pe36_in[31:24]};
                        default: formatted_A0 = '{default: 8'h0};
                    endcase
                end
                default: begin
                    formatted_A0 = '{default: 8'h0};
                end
            endcase
        end
        
        // A1 output (starts 1 cycle later)
        if (cycle_counter >= 1 && state == STREAMING_ROWS) begin
            // Use previous cycle's counters for A1 
            a1_col = (col_counter == 0) ? 6 : col_counter - 1;
            a1_row_group = ((col_counter == 0) && (row_group_counter > 0)) ? row_group_counter - 1 : row_group_counter;
            
            case (a1_row_group)
                2'd0: begin
                    case (a1_col)
                        3'd0: formatted_A1 = '{patch_pe10_in[7:0], patch_pe10_in[15:8], patch_pe10_in[23:16], patch_pe10_in[31:24]};
                        3'd1: formatted_A1 = '{patch_pe11_in[7:0], patch_pe11_in[15:8], patch_pe11_in[23:16], patch_pe11_in[31:24]};
                        3'd2: formatted_A1 = '{patch_pe12_in[7:0], patch_pe12_in[15:8], patch_pe12_in[23:16], patch_pe12_in[31:24]};
                        3'd3: formatted_A1 = '{patch_pe13_in[7:0], patch_pe13_in[15:8], patch_pe13_in[23:16], patch_pe13_in[31:24]};
                        3'd4: formatted_A1 = '{patch_pe14_in[7:0], patch_pe14_in[15:8], patch_pe14_in[23:16], patch_pe14_in[31:24]};
                        3'd5: formatted_A1 = '{patch_pe15_in[7:0], patch_pe15_in[15:8], patch_pe15_in[23:16], patch_pe15_in[31:24]};
                        3'd6: formatted_A1 = '{patch_pe16_in[7:0], patch_pe16_in[15:8], patch_pe16_in[23:16], patch_pe16_in[31:24]};
                        default: formatted_A1 = '{default: 8'h0};
                    endcase
                end
                2'd1: begin
                    case (a1_col)
                        3'd0: formatted_A1 = '{patch_pe20_in[7:0], patch_pe20_in[15:8], patch_pe20_in[23:16], patch_pe20_in[31:24]};
                        3'd1: formatted_A1 = '{patch_pe21_in[7:0], patch_pe21_in[15:8], patch_pe21_in[23:16], patch_pe21_in[31:24]};
                        3'd2: formatted_A1 = '{patch_pe22_in[7:0], patch_pe22_in[15:8], patch_pe22_in[23:16], patch_pe22_in[31:24]};
                        3'd3: formatted_A1 = '{patch_pe23_in[7:0], patch_pe23_in[15:8], patch_pe23_in[23:16], patch_pe23_in[31:24]};
                        3'd4: formatted_A1 = '{patch_pe24_in[7:0], patch_pe24_in[15:8], patch_pe24_in[23:16], patch_pe24_in[31:24]};
                        3'd5: formatted_A1 = '{patch_pe25_in[7:0], patch_pe25_in[15:8], patch_pe25_in[23:16], patch_pe25_in[31:24]};
                        3'd6: formatted_A1 = '{patch_pe26_in[7:0], patch_pe26_in[15:8], patch_pe26_in[23:16], patch_pe26_in[31:24]};
                        default: formatted_A1 = '{default: 8'h0};
                    endcase
                end
                2'd2: begin
                    case (a1_col)
                        3'd0: formatted_A1 = '{patch_pe30_in[7:0], patch_pe30_in[15:8], patch_pe30_in[23:16], patch_pe30_in[31:24]};
                        3'd1: formatted_A1 = '{patch_pe31_in[7:0], patch_pe31_in[15:8], patch_pe31_in[23:16], patch_pe31_in[31:24]};
                        3'd2: formatted_A1 = '{patch_pe32_in[7:0], patch_pe32_in[15:8], patch_pe32_in[23:16], patch_pe32_in[31:24]};
                        3'd3: formatted_A1 = '{patch_pe33_in[7:0], patch_pe33_in[15:8], patch_pe33_in[23:16], patch_pe33_in[31:24]};
                        3'd4: formatted_A1 = '{patch_pe34_in[7:0], patch_pe34_in[15:8], patch_pe34_in[23:16], patch_pe34_in[31:24]};
                        3'd5: formatted_A1 = '{patch_pe35_in[7:0], patch_pe35_in[15:8], patch_pe35_in[23:16], patch_pe35_in[31:24]};
                        3'd6: formatted_A1 = '{patch_pe36_in[7:0], patch_pe36_in[15:8], patch_pe36_in[23:16], patch_pe36_in[31:24]};
                        default: formatted_A1 = '{default: 8'h0};
                    endcase
                end
                2'd3: begin
                    case (a1_col)
                        3'd0: formatted_A1 = '{patch_pe40_in[7:0], patch_pe40_in[15:8], patch_pe40_in[23:16], patch_pe40_in[31:24]};
                        3'd1: formatted_A1 = '{patch_pe41_in[7:0], patch_pe41_in[15:8], patch_pe41_in[23:16], patch_pe41_in[31:24]};
                        3'd2: formatted_A1 = '{patch_pe42_in[7:0], patch_pe42_in[15:8], patch_pe42_in[23:16], patch_pe42_in[31:24]};
                        3'd3: formatted_A1 = '{patch_pe43_in[7:0], patch_pe43_in[15:8], patch_pe43_in[23:16], patch_pe43_in[31:24]};
                        3'd4: formatted_A1 = '{patch_pe44_in[7:0], patch_pe44_in[15:8], patch_pe44_in[23:16], patch_pe44_in[31:24]};
                        3'd5: formatted_A1 = '{patch_pe45_in[7:0], patch_pe45_in[15:8], patch_pe45_in[23:16], patch_pe45_in[31:24]};
                        3'd6: formatted_A1 = '{patch_pe46_in[7:0], patch_pe46_in[15:8], patch_pe46_in[23:16], patch_pe46_in[31:24]};
                        default: formatted_A1 = '{default: 8'h0};
                    endcase
                end
                default: begin
                    formatted_A1 = '{default: 8'h0};
                end
            endcase
        end
        
        // A2 output (starts 2 cycles later) 
        if (cycle_counter >= 2 && state == STREAMING_ROWS) begin
            // Use 2 cycles ago counters for A2
            a2_col = (cycle_counter < 2) ? 0 : ((col_counter >= 2) ? col_counter - 2 : col_counter + 5);
            a2_row_group = (cycle_counter < 2) ? 0 : 
                          ((col_counter >= 2) ? row_group_counter : 
                           (row_group_counter > 0) ? row_group_counter - 1 : 0);
            
            case (a2_row_group)
                2'd0: begin
                    case (a2_col)
                        3'd0: formatted_A2 = '{patch_pe20_in[7:0], patch_pe20_in[15:8], patch_pe20_in[23:16], patch_pe20_in[31:24]};
                        3'd1: formatted_A2 = '{patch_pe21_in[7:0], patch_pe21_in[15:8], patch_pe21_in[23:16], patch_pe21_in[31:24]};
                        3'd2: formatted_A2 = '{patch_pe22_in[7:0], patch_pe22_in[15:8], patch_pe22_in[23:16], patch_pe22_in[31:24]};
                        3'd3: formatted_A2 = '{patch_pe23_in[7:0], patch_pe23_in[15:8], patch_pe23_in[23:16], patch_pe23_in[31:24]};
                        3'd4: formatted_A2 = '{patch_pe24_in[7:0], patch_pe24_in[15:8], patch_pe24_in[23:16], patch_pe24_in[31:24]};
                        3'd5: formatted_A2 = '{patch_pe25_in[7:0], patch_pe25_in[15:8], patch_pe25_in[23:16], patch_pe25_in[31:24]};
                        3'd6: formatted_A2 = '{patch_pe26_in[7:0], patch_pe26_in[15:8], patch_pe26_in[23:16], patch_pe26_in[31:24]};
                        default: formatted_A2 = '{default: 8'h0};
                    endcase
                end
                2'd1: begin
                    case (a2_col)
                        3'd0: formatted_A2 = '{patch_pe30_in[7:0], patch_pe30_in[15:8], patch_pe30_in[23:16], patch_pe30_in[31:24]};
                        3'd1: formatted_A2 = '{patch_pe31_in[7:0], patch_pe31_in[15:8], patch_pe31_in[23:16], patch_pe31_in[31:24]};
                        3'd2: formatted_A2 = '{patch_pe32_in[7:0], patch_pe32_in[15:8], patch_pe32_in[23:16], patch_pe32_in[31:24]};
                        3'd3: formatted_A2 = '{patch_pe33_in[7:0], patch_pe33_in[15:8], patch_pe33_in[23:16], patch_pe33_in[31:24]};
                        3'd4: formatted_A2 = '{patch_pe34_in[7:0], patch_pe34_in[15:8], patch_pe34_in[23:16], patch_pe34_in[31:24]};
                        3'd5: formatted_A2 = '{patch_pe35_in[7:0], patch_pe35_in[15:8], patch_pe35_in[23:16], patch_pe35_in[31:24]};
                        3'd6: formatted_A2 = '{patch_pe36_in[7:0], patch_pe36_in[15:8], patch_pe36_in[23:16], patch_pe36_in[31:24]};
                        default: formatted_A2 = '{default: 8'h0};
                    endcase
                end
                2'd2: begin
                    case (a2_col)
                        3'd0: formatted_A2 = '{patch_pe40_in[7:0], patch_pe40_in[15:8], patch_pe40_in[23:16], patch_pe40_in[31:24]};
                        3'd1: formatted_A2 = '{patch_pe41_in[7:0], patch_pe41_in[15:8], patch_pe41_in[23:16], patch_pe41_in[31:24]};
                        3'd2: formatted_A2 = '{patch_pe42_in[7:0], patch_pe42_in[15:8], patch_pe42_in[23:16], patch_pe42_in[31:24]};
                        3'd3: formatted_A2 = '{patch_pe43_in[7:0], patch_pe43_in[15:8], patch_pe43_in[23:16], patch_pe43_in[31:24]};
                        3'd4: formatted_A2 = '{patch_pe44_in[7:0], patch_pe44_in[15:8], patch_pe44_in[23:16], patch_pe44_in[31:24]};
                        3'd5: formatted_A2 = '{patch_pe45_in[7:0], patch_pe45_in[15:8], patch_pe45_in[23:16], patch_pe45_in[31:24]};
                        3'd6: formatted_A2 = '{patch_pe46_in[7:0], patch_pe46_in[15:8], patch_pe46_in[23:16], patch_pe46_in[31:24]};
                        default: formatted_A2 = '{default: 8'h0};
                    endcase
                end
                2'd3: begin
                    case (a2_col)
                        3'd0: formatted_A2 = '{patch_pe50_in[7:0], patch_pe50_in[15:8], patch_pe50_in[23:16], patch_pe50_in[31:24]};
                        3'd1: formatted_A2 = '{patch_pe51_in[7:0], patch_pe51_in[15:8], patch_pe51_in[23:16], patch_pe51_in[31:24]};
                        3'd2: formatted_A2 = '{patch_pe52_in[7:0], patch_pe52_in[15:8], patch_pe52_in[23:16], patch_pe52_in[31:24]};
                        3'd3: formatted_A2 = '{patch_pe53_in[7:0], patch_pe53_in[15:8], patch_pe53_in[23:16], patch_pe53_in[31:24]};
                        3'd4: formatted_A2 = '{patch_pe54_in[7:0], patch_pe54_in[15:8], patch_pe54_in[23:16], patch_pe54_in[31:24]};
                        3'd5: formatted_A2 = '{patch_pe55_in[7:0], patch_pe55_in[15:8], patch_pe55_in[23:16], patch_pe55_in[31:24]};
                        3'd6: formatted_A2 = '{patch_pe56_in[7:0], patch_pe56_in[15:8], patch_pe56_in[23:16], patch_pe56_in[31:24]};
                        default: formatted_A2 = '{default: 8'h0};
                    endcase
                end
                default: begin
                    formatted_A2 = '{default: 8'h0};
                end
            endcase
        end
        
        // A3 output (starts 3 cycles later)
        if (cycle_counter >= 3 && state == STREAMING_ROWS) begin
            // Use 3 cycles ago counters for A3
            a3_col = (cycle_counter < 3) ? 0 : ((col_counter >= 3) ? col_counter - 3 : col_counter + 4);
            a3_row_group = (cycle_counter < 3) ? 0 : 
                          ((col_counter >= 3) ? row_group_counter : 
                           (row_group_counter > 0) ? row_group_counter - 1 : 0);
            
            case (a3_row_group)
                2'd0: begin
                    case (a3_col)
                        3'd0: formatted_A3 = '{patch_pe30_in[7:0], patch_pe30_in[15:8], patch_pe30_in[23:16], patch_pe30_in[31:24]};
                        3'd1: formatted_A3 = '{patch_pe31_in[7:0], patch_pe31_in[15:8], patch_pe31_in[23:16], patch_pe31_in[31:24]};
                        3'd2: formatted_A3 = '{patch_pe32_in[7:0], patch_pe32_in[15:8], patch_pe32_in[23:16], patch_pe32_in[31:24]};
                        3'd3: formatted_A3 = '{patch_pe33_in[7:0], patch_pe33_in[15:8], patch_pe33_in[23:16], patch_pe33_in[31:24]};
                        3'd4: formatted_A3 = '{patch_pe34_in[7:0], patch_pe34_in[15:8], patch_pe34_in[23:16], patch_pe34_in[31:24]};
                        3'd5: formatted_A3 = '{patch_pe35_in[7:0], patch_pe35_in[15:8], patch_pe35_in[23:16], patch_pe35_in[31:24]};
                        3'd6: formatted_A3 = '{patch_pe36_in[7:0], patch_pe36_in[15:8], patch_pe36_in[23:16], patch_pe36_in[31:24]};
                        default: formatted_A3 = '{default: 8'h0};
                    endcase
                end
                2'd1: begin
                    case (a3_col)
                        3'd0: formatted_A3 = '{patch_pe40_in[7:0], patch_pe40_in[15:8], patch_pe40_in[23:16], patch_pe40_in[31:24]};
                        3'd1: formatted_A3 = '{patch_pe41_in[7:0], patch_pe41_in[15:8], patch_pe41_in[23:16], patch_pe41_in[31:24]};
                        3'd2: formatted_A3 = '{patch_pe42_in[7:0], patch_pe42_in[15:8], patch_pe42_in[23:16], patch_pe42_in[31:24]};
                        3'd3: formatted_A3 = '{patch_pe43_in[7:0], patch_pe43_in[15:8], patch_pe43_in[23:16], patch_pe43_in[31:24]};
                        3'd4: formatted_A3 = '{patch_pe44_in[7:0], patch_pe44_in[15:8], patch_pe44_in[23:16], patch_pe44_in[31:24]};
                        3'd5: formatted_A3 = '{patch_pe45_in[7:0], patch_pe45_in[15:8], patch_pe45_in[23:16], patch_pe45_in[31:24]};
                        3'd6: formatted_A3 = '{patch_pe46_in[7:0], patch_pe46_in[15:8], patch_pe46_in[23:16], patch_pe46_in[31:24]};
                        default: formatted_A3 = '{default: 8'h0};
                    endcase
                end
                2'd2: begin
                    case (a3_col)
                        3'd0: formatted_A3 = '{patch_pe50_in[7:0], patch_pe50_in[15:8], patch_pe50_in[23:16], patch_pe50_in[31:24]};
                        3'd1: formatted_A3 = '{patch_pe51_in[7:0], patch_pe51_in[15:8], patch_pe51_in[23:16], patch_pe51_in[31:24]};
                        3'd2: formatted_A3 = '{patch_pe52_in[7:0], patch_pe52_in[15:8], patch_pe52_in[23:16], patch_pe52_in[31:24]};
                        3'd3: formatted_A3 = '{patch_pe53_in[7:0], patch_pe53_in[15:8], patch_pe53_in[23:16], patch_pe53_in[31:24]};
                        3'd4: formatted_A3 = '{patch_pe54_in[7:0], patch_pe54_in[15:8], patch_pe54_in[23:16], patch_pe54_in[31:24]};
                        3'd5: formatted_A3 = '{patch_pe55_in[7:0], patch_pe55_in[15:8], patch_pe55_in[23:16], patch_pe55_in[31:24]};
                        3'd6: formatted_A3 = '{patch_pe56_in[7:0], patch_pe56_in[15:8], patch_pe56_in[23:16], patch_pe56_in[31:24]};
                        default: formatted_A3 = '{default: 8'h0};
                    endcase
                end
                2'd3: begin
                    case (a3_col)
                        3'd0: formatted_A3 = '{patch_pe60_in[7:0], patch_pe60_in[15:8], patch_pe60_in[23:16], patch_pe60_in[31:24]};
                        3'd1: formatted_A3 = '{patch_pe61_in[7:0], patch_pe61_in[15:8], patch_pe61_in[23:16], patch_pe61_in[31:24]};
                        3'd2: formatted_A3 = '{patch_pe62_in[7:0], patch_pe62_in[15:8], patch_pe62_in[23:16], patch_pe62_in[31:24]};
                        3'd3: formatted_A3 = '{patch_pe63_in[7:0], patch_pe63_in[15:8], patch_pe63_in[23:16], patch_pe63_in[31:24]};
                        3'd4: formatted_A3 = '{patch_pe64_in[7:0], patch_pe64_in[15:8], patch_pe64_in[23:16], patch_pe64_in[31:24]};
                        3'd5: formatted_A3 = '{patch_pe65_in[7:0], patch_pe65_in[15:8], patch_pe65_in[23:16], patch_pe65_in[31:24]};
                        3'd6: formatted_A3 = '{patch_pe66_in[7:0], patch_pe66_in[15:8], patch_pe66_in[23:16], patch_pe66_in[31:24]};
                        default: formatted_A3 = '{default: 8'h0};
                    endcase
                end
                default: begin
                    formatted_A3 = '{default: 8'h0};
                end
            endcase
        end
    end
    
    assign formatted_data_valid = (state == STREAMING_ROWS) && patches_valid;
    assign all_cols_sent = (state == IDLE) && (cycle_counter == 0);
    assign next_block = all_cols_sent; // Request next block when all cycles are sent

endmodule 
