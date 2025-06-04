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
    logic [2:0] col_counter;      // 0-6 for 7 columns (Not directly used for indexing anymore)
    logic [1:0] row_group_counter; // 0-3 for 4 row groups (Not directly used for indexing anymore)
    logic [4:0] cycle_counter;    // Overall cycle counter for staggered timing
    
    // State machine for complete spatial streaming
    typedef enum logic [1:0] {
        IDLE,
        STREAMING_ROWS
    } state_t;
    state_t state;
    
    // Complete spatial streaming state machine - auto-advance every clock cycle
    always_ff @(posedge clk) begin
        if (reset) begin
            col_counter <= 0; // Still used by FSM for overall 7-col cycle
            row_group_counter <= 0; // Still used by FSM for overall 4-row_group cycle
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
                        
                        // Main FSM counters track overall progress, not individual lane data here
                        if (col_counter == 6) begin 
                            col_counter <= 0;
                            if (row_group_counter == 3) begin
                                // All 4 row groups completed (28 cycles total + 3 extra for staggering)
                                if (cycle_counter >= 30) begin // 28 data cycles + 3 stagger cycles to complete A3
                                    row_group_counter <= 0;
                                    // cycle_counter <= 0; // cycle_counter reset by FSM logic
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
        // Default assignments for outputs
        formatted_A0 = '{default: 8'h0};
        formatted_A1 = '{default: 8'h0};
        formatted_A2 = '{default: 8'h0};
        formatted_A3 = '{default: 8'h0};

        // Intermediate calculation signals - declared and defaulted here
        logic [2:0] current_a0_col_calc;
        logic [1:0] current_a0_row_group_calc;
        
        logic [4:0] effective_cycle_A1_calc;
        logic [2:0] current_a1_col_calc;
        logic [1:0] current_a1_row_group_calc;

        logic [4:0] effective_cycle_A2_calc;
        logic [2:0] current_a2_col_calc;
        logic [1:0] current_a2_row_group_calc;

        logic [4:0] effective_cycle_A3_calc;
        logic [2:0] current_a3_col_calc;
        logic [1:0] current_a3_row_group_calc;

        // Default assignments for calculation signals to prevent latches
        current_a0_col_calc = 3'b0;
        current_a0_row_group_calc = 2'b0;

        effective_cycle_A1_calc = 5'b0;
        current_a1_col_calc = 3'b0;
        current_a1_row_group_calc = 2'b0;

        effective_cycle_A2_calc = 5'b0;
        current_a2_col_calc = 3'b0;
        current_a2_row_group_calc = 2'b0;

        effective_cycle_A3_calc = 5'b0;
        current_a3_col_calc = 3'b0;
        current_a3_row_group_calc = 2'b0;
        
        // A0 output (active for cycle_counter 0-27)
        if (state == STREAMING_ROWS && cycle_counter <= 27) begin
            current_a0_col_calc = 3'(cycle_counter % 7);
            current_a0_row_group_calc = 2'(cycle_counter / 7);

            case (current_a0_row_group_calc)
                2'd0: begin // Row Group 0 for A0: maps to PE rows 0-3
                    case (current_a0_col_calc)
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
                2'd1: begin // Row Group 1 for A0: maps to PE rows 1-4
                    case (current_a0_col_calc)
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
                2'd2: begin // Row Group 2 for A0: maps to PE rows 2-5
                    case (current_a0_col_calc)
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
                2'd3: begin // Row Group 3 for A0: maps to PE rows 3-6
                    case (current_a0_col_calc)
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
        
        // A1 output (active for cycle_counter 1-28)
        if (state == STREAMING_ROWS && cycle_counter >= 1 && cycle_counter <= 28) begin
            effective_cycle_A1_calc = cycle_counter - 1;
            current_a1_col_calc = 3'(effective_cycle_A1_calc % 7);
            current_a1_row_group_calc = 2'(effective_cycle_A1_calc / 7);
            
            case (current_a1_row_group_calc)
                2'd0: begin // Row Group 0 for A1: maps to PE rows 1-4
                    case (current_a1_col_calc)
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
                2'd1: begin // Row Group 1 for A1: maps to PE rows 2-5
                    case (current_a1_col_calc)
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
                2'd2: begin // Row Group 2 for A1: maps to PE rows 3-6
                    case (current_a1_col_calc)
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
                2'd3: begin // Row Group 3 for A1: maps to PE rows 4-7
                    case (current_a1_col_calc)
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
        
        // A2 output (active for cycle_counter 2-29) 
        if (state == STREAMING_ROWS && cycle_counter >= 2 && cycle_counter <= 29) begin
            effective_cycle_A2_calc = cycle_counter - 2;
            current_a2_col_calc = 3'(effective_cycle_A2_calc % 7);
            current_a2_row_group_calc = 2'(effective_cycle_A2_calc / 7);
            
            case (current_a2_row_group_calc)
                2'd0: begin // Row Group 0 for A2: maps to PE rows 2-5
                    case (current_a2_col_calc)
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
                2'd1: begin // Row Group 1 for A2: maps to PE rows 3-6
                    case (current_a2_col_calc)
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
                2'd2: begin // Row Group 2 for A2: maps to PE rows 4-7
                    case (current_a2_col_calc)
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
                2'd3: begin // Row Group 3 for A2: maps to PE rows 5-8 (patch_pe5X_in)
                    case (current_a2_col_calc)
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
        
        // A3 output (active for cycle_counter 3-30)
        if (state == STREAMING_ROWS && cycle_counter >= 3 && cycle_counter <= 30) begin
            effective_cycle_A3_calc = cycle_counter - 3;
            current_a3_col_calc = 3'(effective_cycle_A3_calc % 7);
            current_a3_row_group_calc = 2'(effective_cycle_A3_calc / 7);
            
            case (current_a3_row_group_calc)
                2'd0: begin // Row Group 0 for A3: maps to PE rows 3-6
                    case (current_a3_col_calc)
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
                2'd1: begin // Row Group 1 for A3: maps to PE rows 4-7
                    case (current_a3_col_calc)
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
                2'd2: begin // Row Group 2 for A3: maps to PE rows 5-8 (patch_pe5X_in)
                    case (current_a3_col_calc)
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
                2'd3: begin // Row Group 3 for A3: maps to PE rows 6-9 (patch_pe6X_in)
                    case (current_a3_col_calc)
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
