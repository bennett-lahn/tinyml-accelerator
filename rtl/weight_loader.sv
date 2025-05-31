`include "sys_types.svh"

// Weight Loader for Systolic Array Direct Convolution
// Responsible for timing weight loading from weight ROM to systolic array
// Weight ROM indexed in channel, row, col order: (row,col,input_channel)
// Indexing: (0,0,0), (0,0,1), ..., (0,1,0), (0,1,1), ..., (1,0,0), (1,0,1)...
// Manages 4x4x4 buffer for (row,col,ch) and times weight delivery per column

module weight_loader #(
    parameter int MAX_NUM_CH = 64,              // Maximum number of input channels
    parameter int CH_BITS = $clog2(MAX_NUM_CH+1), // Bits to hold channel number
    parameter int KERNEL_SIZE = 4,              // 4x4 kernel size
    parameter int SA_N = 4,                     // Systolic array dimension (4x4)
    parameter int VECTOR_WIDTH = 4,             // Vector width for each systolic array input
    parameter int ROM_DEPTH = 16384,            // Weight ROM depth
    parameter int ROM_ADDR_BITS = $clog2(ROM_DEPTH),
    
    // Layer configuration parameters
    parameter int NUM_LAYERS = 6,
    parameter int CONV_IN_C [NUM_LAYERS] = '{1,  8,  16,  32, 256,64},
    parameter int CONV_OUT_C [NUM_LAYERS] = '{8,  16, 32,  64, 64, 10}
)(
    input  logic clk,
    input  logic reset,
    input  logic start,                         // Start weight loading sequence
    input  logic stall,                         // Stall signal from system
    
    // Configuration inputs
    input  logic [$clog2(MAX_NUM_CH)-1:0] output_channel_idx, // Current output channel being processed
    input  logic [$clog2(NUM_LAYERS)-1:0] current_layer_idx, // Current layer index
    
    // Weight ROM interface
    output logic weight_rom_read_enable,
    output logic [ROM_ADDR_BITS-1:0] weight_rom_addr,
    input  logic [31:0] weight_rom_data0,       // Position (row,3) for 4 input channels (furthest from address)
    input  logic [31:0] weight_rom_data1,       // Position (row,2) for 4 input channels
    input  logic [31:0] weight_rom_data2,       // Position (row,1) for 4 input channels
    input  logic [31:0] weight_rom_data3,       // Position (row,0) for 4 input channels (closest to address)
    
    // Systolic array weight outputs (B matrix inputs)
    output int8_t B0 [VECTOR_WIDTH],        // Column 0 weights
    output int8_t B1 [VECTOR_WIDTH],        // Column 1 weights  
    output int8_t B2 [VECTOR_WIDTH],        // Column 2 weights
    output int8_t B3 [VECTOR_WIDTH],        // Column 3 weights
    
    // Status outputs
    output logic idle,                          // High when not actively loading weights
    output logic weight_load_complete           // High when all weights for current tile loaded
);

    // Main FSM state encoding
    typedef enum logic [1:0] {
        MAIN_IDLE,
        MAIN_RUNNING,
        MAIN_DONE
    } main_state_t;

    // Column FSM state encoding
    typedef enum logic [2:0] {
        COL_IDLE,
        COL_DELAY,
        COL_WEIGHT_PHASE,
        COL_ZERO_PHASE,
        COL_DONE
    } col_state_t;

    main_state_t main_current_state, main_next_state;
    col_state_t col_state [0:3];  // State for each column
    col_state_t col_next_state [0:3];  // Next state for each column

    // 4x4x4 weight buffer: [count][channel_group] where count is 0-15 in row-major order
    // Each element stores 4 int8 weights for 4 input channels
    int8_t weight_buffer [16][VECTOR_WIDTH];
    
    // Main FSM counters and control signals
    logic [CH_BITS-1:0] current_input_channel_group;   // Current group of 4 input channels being loaded
    logic [CH_BITS-1:0] max_channel_groups;            // Total number of channel groups needed
    logic [$clog2(KERNEL_SIZE)-1:0] buffer_row; // For loading buffer
    logic all_columns_done;
    logic buffer_loading_active;                        // True when we're actively loading buffer
    
    // Column-specific counters and control
    logic [1:0] col_delay_count [0:3];       // Delay counter for each column before starting
    logic [2:0] col_cycle_count [0:3];       // Cycle counter within weight/zero phases for each column
    logic [3:0] col_weight_position [0:3];   // Current weight position in row-major order (0-15) for each column
    logic col_active [0:3];                  // Whether each column is actively sending weights
    
    // Channel group transition control
    logic transitioning_to_next_channel_group;      // Signal to indicate we're transitioning to next input channel group
    
    // ROM control signals
    logic rom_data_valid;                           // ROM data is valid this cycle (internal use)
    
    // ROM data valid register - delayed version of weight_rom_read_enable
    always_ff @(posedge clk) begin
        if (reset) begin
            rom_data_valid <= 1'b0;
        end else begin
            rom_data_valid <= weight_rom_read_enable;
        end
    end
    
    // Function to calculate cumulative layer base offsets at compile time
    function automatic logic [ROM_ADDR_BITS-1:0] calculate_layer_offset(int layer_idx);
        logic [ROM_ADDR_BITS-1:0] offset;
        offset = 0;
        for (int i = 0; i < layer_idx; i++) begin
            offset = offset + ROM_ADDR_BITS'(CONV_OUT_C[i] * CONV_IN_C[i]);
        end
        return offset;
    endfunction

    // Pre-calculated layer base offsets using generate
    logic [ROM_ADDR_BITS-1:0] LAYER_BASE_OFFSETS [NUM_LAYERS];
    
    generate
        genvar layer_gen;
        // Layer 0 has no base offset
        assign LAYER_BASE_OFFSETS[0] = 0;
        for (layer_gen = 1; layer_gen < NUM_LAYERS; layer_gen++) begin : gen_layer_offsets
            assign LAYER_BASE_OFFSETS[layer_gen] = calculate_layer_offset(layer_gen);
        end
    endgenerate
    
    // Check if all columns are done
    always_comb begin
        all_columns_done = 1'b1;
        for (int i = 0; i < 4; i++) begin
            if (col_state[i] != COL_DONE) begin
                all_columns_done = 1'b0;
            end
        end
    end
    
    // Check if all columns have completed their zero phase (for channel group transitions)
    logic all_columns_completed_zero_phase;
    always_comb begin
        all_columns_completed_zero_phase = 1'b1;
        for (int i = 0; i < 4; i++) begin
            if (col_state[i] == COL_ZERO_PHASE && col_cycle_count[i] != 3'd2) begin
                all_columns_completed_zero_phase = 1'b0;
            end
            if (col_state[i] != COL_ZERO_PHASE && col_state[i] != COL_DONE) begin
                all_columns_completed_zero_phase = 1'b0;
            end
        end
        // Only trigger when at least one column is in zero phase
        if (!(col_state[0] == COL_ZERO_PHASE || col_state[1] == COL_ZERO_PHASE || 
              col_state[2] == COL_ZERO_PHASE || col_state[3] == COL_ZERO_PHASE)) begin
            all_columns_completed_zero_phase = 1'b0;
        end
    end
    
    // Calculate maximum channel groups (ceiling division by 4) for current layer
    always_comb begin
        max_channel_groups = CH_BITS'((CONV_IN_C[current_layer_idx] + VECTOR_WIDTH - 1) / VECTOR_WIDTH);
    end
    
    // Calculate weight ROM address based on row-major indexing with proper layer offset calculation
    // ROM layout: All weights for layer 0, then all weights for layer 1, etc.
    // Within each layer: Each individual output filter, then input channel groups (4 addresses per group), then rows
    // Each input channel group has 4 consecutive addresses, one per row
    // Each ROM read returns one complete row (4 columns) for 4 input channels
    always_comb begin
        logic [$clog2(ROM_DEPTH)-1:0] layer_base_offset;
        logic [$clog2(ROM_DEPTH)-1:0] within_layer_offset;
        logic [$clog2(MAX_NUM_CH+1)-1:0] current_layer_input_channel_groups;
        
        // Calculate input channel groups for current layer
        current_layer_input_channel_groups = ($clog2(MAX_NUM_CH+1))'((CONV_IN_C[current_layer_idx] + VECTOR_WIDTH - 1) / VECTOR_WIDTH);
        
        // Calculate base address offset for current layer (sum of all previous layers' weights)
        layer_base_offset = LAYER_BASE_OFFSETS[current_layer_idx];
        
        // Calculate offset within current layer
        // Address pattern: Each input channel group gets 4 consecutive addresses (one per row)
        // within_layer_offset = (output_filter_idx * 4_rows * input_channel_groups) + (input_channel_group * 4_rows) + row_position
        within_layer_offset = (output_channel_idx * 4 * current_layer_input_channel_groups) +
                             (current_input_channel_group * 4) +
                             ($clog2(ROM_DEPTH))'(buffer_row);
        
        // Final ROM address = layer base offset + within layer offset
        weight_rom_addr = layer_base_offset + within_layer_offset;
    end
    
    // Main state register
    always_ff @(posedge clk) begin
        if (reset) begin
            main_current_state <= MAIN_IDLE;
        end else if (!stall) begin
            main_current_state <= main_next_state;
        end
    end
    
    // Column state registers
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < 4; i++) begin
                col_state[i] <= COL_IDLE;
            end
        end else if (!stall) begin
            for (int i = 0; i < 4; i++) begin
                col_state[i] <= col_next_state[i];
            end
        end
    end
    
    // Main FSM counter management
    always_ff @(posedge clk) begin
        if (reset) begin
            current_input_channel_group <= 'd0;
            buffer_row <= 'd0;
        end else if (!stall) begin
            case (main_current_state)
                MAIN_IDLE: begin
                    if (start) begin
                        current_input_channel_group <= 'd0;
                        // Don't reset buffer_row here - only reset when transitioning to next channel
                    end
                end
                
                MAIN_RUNNING: begin
                    // Advance to next channel group when transitioning
                    // Special case for first layer with only 1 input channel
                    if (current_layer_idx == 0) begin
                        buffer_row <= ($clog2(KERNEL_SIZE))'(KERNEL_SIZE - 1);
                    end else begin
                        if (transitioning_to_next_channel_group) begin
                            current_input_channel_group <= current_input_channel_group + 1'd1;
                            buffer_row <= 'd0;  // Reset buffer_row only for next channel group
                        end else if (rom_data_valid) begin
                        // Increment buffer_row when ROM is being read (loading weights)
                            if (buffer_row == ($clog2(KERNEL_SIZE))'(KERNEL_SIZE - 1)) begin
                                // All 4 rows loaded for current channel group, keep at max
                                // Don't reset here - only reset when transitioning to next channel
                                buffer_row <= buffer_row; // Stay at max value
                            end else begin
                                buffer_row <= buffer_row + 1'd1;
                            end
                        end
                    end
                end
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Column-specific counter management
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < 4; i++) begin
                col_delay_count[i] <= 'd0;
                col_cycle_count[i] <= 'd0;
                col_weight_position[i] <= 'd0;
            end
        end else if (!stall) begin
            // Reset all column counters when transitioning to next channel group
            if (transitioning_to_next_channel_group) begin
                for (int i = 0; i < 4; i++) begin
                    col_delay_count[i] <= 'd0;
                    col_cycle_count[i] <= 'd0;
                    col_weight_position[i] <= 'd0;
                end
            end else begin
                for (int i = 0; i < 4; i++) begin
                    case (col_state[i])
                        COL_IDLE: begin
                            if (main_current_state == MAIN_RUNNING) begin
                                col_delay_count[i] <= 'd0;
                                col_cycle_count[i] <= 'd0;
                                col_weight_position[i] <= 'd0;
                            end
                        end
                        
                        COL_DELAY: begin
                            col_delay_count[i] <= col_delay_count[i] + 1'd1;
                        end
                        
                        COL_WEIGHT_PHASE: begin
                            col_cycle_count[i] <= col_cycle_count[i] + 1'd1;
                            // Advance weight position each cycle in row-major order
                            if (col_weight_position[i] < 4'd15) begin
                                col_weight_position[i] <= col_weight_position[i] + 1'd1;
                            end
                            
                            // Reset cycle count after 4 cycles, and reset position if we've completed all 16
                            if (col_cycle_count[i] == 3'd3) begin // After 4 cycles
                                col_cycle_count[i] <= 'd0;
                            end
                        end
                        
                        COL_ZERO_PHASE: begin
                            col_cycle_count[i] <= col_cycle_count[i] + 1'd1;
                            if (col_cycle_count[i] == 3'd2) begin // After 3 cycles
                                col_cycle_count[i] <= 'd0;
                                // Don't reset position - it continues from where it left off
                            end
                        end
                        
                        default: begin
                            // Keep current values
                        end
                    endcase
                end
            end
        end
    end
    
    // Buffer loading and passthrough logic
    always_ff @(posedge clk) begin
        if (reset) begin
            // Clear weight buffer
            for (int r = 0; r < 16; r++) begin
                for (int ch = 0; ch < VECTOR_WIDTH; ch++) begin
                    weight_buffer[r][ch] <= 8'd0;
                end
            end
        end else if (!stall && main_current_state == MAIN_RUNNING && rom_data_valid) begin
            // Load weights from ROM data into buffer
            // Store in buffer using row-major indexing for systolic array output
            
            // Check if this is the special case: first layer with only 1 input channel
            if (current_layer_idx == 0 && CONV_IN_C[0] == 1) begin
                // Special case: First layer with 1 input channel
                // ROM data layout: rom_data3 = entire row 0, rom_data2 = entire row 1, etc.
                // rom_data3: (0,0,0), (0,1,0), (0,2,0), (0,3,0)
                // rom_data2: (1,0,0), (1,1,0), (1,2,0), (1,3,0)
                // rom_data1: (2,0,0), (2,1,0), (2,2,0), (2,3,0)
                // rom_data0: (3,0,0), (3,1,0), (3,2,0), (3,3,0)
                
                // Extract weights for row 0 (from rom_data3)
                weight_buffer[0][0] <= weight_rom_data3[31:24];  // (0,0,0)
                weight_buffer[1][0] <= weight_rom_data3[23:16];  // (0,1,0)
                weight_buffer[2][0] <= weight_rom_data3[15:8];   // (0,2,0)
                weight_buffer[3][0] <= weight_rom_data3[7:0];    // (0,3,0)
                
                // Extract weights for row 1 (from rom_data2)
                weight_buffer[4][0] <= weight_rom_data2[31:24];  // (1,0,0)
                weight_buffer[5][0] <= weight_rom_data2[23:16];  // (1,1,0)
                weight_buffer[6][0] <= weight_rom_data2[15:8];   // (1,2,0)
                weight_buffer[7][0] <= weight_rom_data2[7:0];    // (1,3,0)
                
                // Extract weights for row 2 (from rom_data1)
                weight_buffer[8][0] <= weight_rom_data1[31:24];  // (2,0,0)
                weight_buffer[9][0] <= weight_rom_data1[23:16];  // (2,1,0)
                weight_buffer[10][0] <= weight_rom_data1[15:8];  // (2,2,0)
                weight_buffer[11][0] <= weight_rom_data1[7:0];   // (2,3,0)
                
                // Extract weights for row 3 (from rom_data0)
                weight_buffer[12][0] <= weight_rom_data0[31:24]; // (3,0,0)
                weight_buffer[13][0] <= weight_rom_data0[23:16]; // (3,1,0)
                weight_buffer[14][0] <= weight_rom_data0[15:8];  // (3,2,0)
                weight_buffer[15][0] <= weight_rom_data0[7:0];   // (3,3,0)
                
                // Set unused input channels (1, 2, 3) to zero
                for (int pos = 0; pos < 16; pos++) begin
                    weight_buffer[pos][1] <= 8'd0;
                    weight_buffer[pos][2] <= 8'd0;
                    weight_buffer[pos][3] <= 8'd0;
                end
                
            end else begin
                // Regular case: Multiple input channels (divisible by 4)
                // ROM returns one complete row (4 columns) for 4 input channels
                // rom_data3 = (row,0) for input channels 0,1,2,3
                // rom_data2 = (row,1) for input channels 0,1,2,3
                // rom_data1 = (row,2) for input channels 0,1,2,3
                // rom_data0 = (row,3) for input channels 0,1,2,3
                
                // Column 0 data for all 4 input channels (position 0, 4, 8, 12)

                weight_buffer[buffer_row*4 + 0][0] <= weight_rom_data3[31:24]; // Input channel 0, col 0
                weight_buffer[buffer_row*4 + 0][1] <= weight_rom_data3[23:16]; // Input channel 1, col 0
                weight_buffer[buffer_row*4 + 0][2] <= weight_rom_data3[15:8];  // Input channel 2, col 0
                weight_buffer[buffer_row*4 + 0][3] <= weight_rom_data3[7:0];   // Input channel 3, col 0
                
                // Column 1 data for all 4 input channels (position 1, 5, 9, 13)
                weight_buffer[buffer_row*4 + 1][0] <= weight_rom_data2[31:24]; // Input channel 0, col 1
                weight_buffer[buffer_row*4 + 1][1] <= weight_rom_data2[23:16]; // Input channel 1, col 1
                weight_buffer[buffer_row*4 + 1][2] <= weight_rom_data2[15:8];  // Input channel 2, col 1
                weight_buffer[buffer_row*4 + 1][3] <= weight_rom_data2[7:0];   // Input channel 3, col 1
                
                // Column 2 data for all 4 input channels (position 2, 6, 10, 14)
                weight_buffer[buffer_row*4 + 2][0] <= weight_rom_data1[31:24]; // Input channel 0, col 2
                weight_buffer[buffer_row*4 + 2][1] <= weight_rom_data1[23:16]; // Input channel 1, col 2
                weight_buffer[buffer_row*4 + 2][2] <= weight_rom_data1[15:8];  // Input channel 2, col 2
                weight_buffer[buffer_row*4 + 2][3] <= weight_rom_data1[7:0];   // Input channel 3, col 2
                
                // Column 3 data for all 4 input channels (position 3, 7, 11, 15)
                weight_buffer[buffer_row*4 + 3][0] <= weight_rom_data0[31:24]; // Input channel 0, col 3
                weight_buffer[buffer_row*4 + 3][1] <= weight_rom_data0[23:16]; // Input channel 1, col 3
                weight_buffer[buffer_row*4 + 3][2] <= weight_rom_data0[15:8];  // Input channel 2, col 3
                weight_buffer[buffer_row*4 + 3][3] <= weight_rom_data0[7:0];   // Input channel 3, col 3
            end
        end
    end

    always_ff @(posedge clk) begin
        $display("=== Weight Buffer Contents ===");
        for (int row = 0; row < 4; row++) begin
            for (int col = 0; col < 4; col++) begin
                int pos = row * 4 + col;
                $display("Pos[%0d] (r%0d,c%0d): Ch[0]=%0d Ch[1]=%0d Ch[2]=%0d Ch[3]=%0d",
                    pos, row, col,
                    $signed(weight_buffer[pos][0]),
                    $signed(weight_buffer[pos][1]),
                    $signed(weight_buffer[pos][2]),
                    $signed(weight_buffer[pos][3]));
            end
        end
        $display("===========================");
    end
    
    // Check if buffer is fully loaded (all rows loaded for current channel group)
    logic buffer_fully_loaded;
    assign buffer_fully_loaded = (buffer_row == ($clog2(KERNEL_SIZE))'(KERNEL_SIZE-1));
    
    // Passthrough control - simplified, only for column 0 when buffer not ready
    logic use_passthrough_col0;
    assign use_passthrough_col0 = (col_state[0] == COL_WEIGHT_PHASE) && !buffer_fully_loaded;
    
    // Column active signals
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            col_active[i] = (col_state[i] == COL_WEIGHT_PHASE);
        end
    end
    
    // Detect transition to next channel group
    always_comb begin
        logic all_weights_at_max;
        logic all_columns_inactive;
        
        // Check if all weight positions are at maximum (15)
        all_weights_at_max = 1'b1;
        for (int i = 0; i < 4; i++) begin
            if (col_weight_position[i] != 4'd15) begin
                all_weights_at_max = 1'b0;
            end
        end
        // Check if all columns are inactive
        all_columns_inactive = 1'b1;
        for (int i = 0; i < 4; i++) begin
            if (col_active[i] != 0) begin
                all_columns_inactive = 1'b0;
            end
        end
        // Combine conditions for transition
        transitioning_to_next_channel_group = all_weights_at_max && 
                                            all_columns_inactive && 
                                            (current_input_channel_group < max_channel_groups - 1);
    end
    
    // Main FSM next state logic
    always_comb begin
        main_next_state = main_current_state;
        
        case (main_current_state)
            MAIN_IDLE: begin
                if (start) begin
                    main_next_state = MAIN_RUNNING;
                end
            end
            
            MAIN_RUNNING: begin
                if (all_columns_done) begin
                    main_next_state = MAIN_DONE;
                end else if (all_columns_completed_zero_phase && 
                           current_input_channel_group < max_channel_groups - 1) begin
                    // Need to load next channel group
                    main_next_state = MAIN_RUNNING;
                end
            end
            
            MAIN_DONE: begin
                if (start) begin // Restart on new start signal
                    main_next_state = MAIN_RUNNING;
                end
            end
            
            default: begin
                main_next_state = MAIN_IDLE;
            end
        endcase
    end
    
    // Column FSM next state logic
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            col_next_state[i] = col_state[i];
            
            // Force all columns to COL_IDLE when transitioning to next channel group
            if (transitioning_to_next_channel_group) begin
                col_next_state[i] = COL_IDLE;
            end else begin
                case (col_state[i])
                    COL_IDLE: begin
                        if (main_current_state == MAIN_RUNNING) begin
                            if (i == 0) begin
                                // Column 0 has no initial delay, go directly to weight phase
                                col_next_state[i] = COL_WEIGHT_PHASE;
                            end else begin
                                // Columns 1-3 need initial delays
                                col_next_state[i] = COL_DELAY;
                            end
                        end
                    end
                    
                    COL_DELAY: begin
                        // Each column has a different delay: col 1=1 cycle, col 2=2 cycles, col 3=3 cycles
                        // col_delay_count starts at 0 and increments each cycle in COL_DELAY state
                        // So we transition when count reaches (i-1) to get i cycles of delay
                        if (col_delay_count[i] == 2'(i - 1)) begin
                            col_next_state[i] = COL_WEIGHT_PHASE;
                        end
                    end
                    
                    COL_WEIGHT_PHASE: begin
                        // Send weights for 4 cycles, then go to zero phase
                        if (col_cycle_count[i] == 3'd3) begin // After 4 cycles
                            col_next_state[i] = COL_ZERO_PHASE;
                        end
                    end
                    
                    COL_ZERO_PHASE: begin
                        // Send zeros for 3 cycles
                        if (col_cycle_count[i] == 3'd2) begin
                            if (col_weight_position[i] == 4'd15) begin // All 16 positions complete (reset to 0)
                                if (current_input_channel_group == max_channel_groups - 1) begin
                                    col_next_state[i] = COL_DONE; // All channel groups processed
                                end else begin
                                    // Transition to next channel group will be handled by transitioning_to_next_channel_group
                                    col_next_state[i] = COL_ZERO_PHASE; // Stay in zero phase until transition
                                end
                            end else begin
                                col_next_state[i] = COL_WEIGHT_PHASE; // Continue with next 4 positions
                            end
                        end
                    end
                    
                    COL_DONE: begin
                        if (start) begin // Restart on new start signal
                            col_next_state[i] = COL_IDLE;
                        end
                    end
                    
                    default: begin
                        col_next_state[i] = COL_IDLE;
                    end
                endcase
            end
        end
    end
    
    // Signals used for special index case
    logic [1:0] row_pos;
    logic [1:0] col_pos;
    logic [31:0] selected_rom_data;

    // Output weight assignment
    always_comb begin
        // Default to zeros
        for (int i = 0; i < VECTOR_WIDTH; i++) begin
            B0[i] = 8'd0;
            B1[i] = 8'd0;
            B2[i] = 8'd0;
            B3[i] = 8'd0;
        end
        // Initialize to prevent latches
        row_pos = 2'b00;
        col_pos = 2'b00;
        selected_rom_data = 32'h0;
        
        // Column 0: Use passthrough when buffer not ready, otherwise use buffer
        if (col_state[0] == COL_WEIGHT_PHASE) begin
            if (use_passthrough_col0 && rom_data_valid) begin
                // Use ROM data directly for immediate response when buffer not ready
                
                // Check if this is the special case: first layer with only 1 input channel
                if (current_layer_idx == 0 && CONV_IN_C[0] == 1) begin
                    // Special case: ROM data contains entire rows across columns for single input channel
                    // Select the appropriate ROM data based on row position from weight position
                    
                    row_pos = 2'(col_weight_position[0] / 4);  // Extract row from row-major position
                    col_pos = 2'(col_weight_position[0] % 4);  // Extract column from row-major position
                    
                    // Select ROM data based on row
                    case (row_pos)
                        2'b00: selected_rom_data = weight_rom_data3; // Row 0
                        2'b01: selected_rom_data = weight_rom_data2; // Row 1  
                        2'b10: selected_rom_data = weight_rom_data1; // Row 2
                        2'b11: selected_rom_data = weight_rom_data0; // Row 3
                    endcase
                    
                    // Extract weight for current column position and input channel 0
                    case (col_pos)
                        2'b00: B0[0] = selected_rom_data[31:24]; // Column 0
                        2'b01: B0[0] = selected_rom_data[23:16]; // Column 1
                        2'b10: B0[0] = selected_rom_data[15:8];  // Column 2
                        2'b11: B0[0] = selected_rom_data[7:0];   // Column 3
                    endcase
                    
                    // Set unused input channels to zero
                    B0[1] = 8'd0;
                    B0[2] = 8'd0;
                    B0[3] = 8'd0;
                    
                end else begin
                    // Regular case: Multiple input channels
                    // Select the appropriate ROM data based on column position from weight position
                    col_pos = 2'(col_weight_position[0] % 4);  // Extract column from row-major position
                    
                    case (col_pos)
                        2'b00: selected_rom_data = weight_rom_data3; // Column 0
                        2'b01: selected_rom_data = weight_rom_data2; // Column 1  
                        2'b10: selected_rom_data = weight_rom_data1; // Column 2
                        2'b11: selected_rom_data = weight_rom_data0; // Column 3
                    endcase
                    
                    B0[0] = selected_rom_data[31:24];   // Input channel 0
                    B0[1] = selected_rom_data[23:16];   // Input channel 1
                    B0[2] = selected_rom_data[15:8];    // Input channel 2
                    B0[3] = selected_rom_data[7:0];     // Input channel 3
                end
            end else begin
                // Use buffered data - get weights for current position
                B0 = weight_buffer[col_weight_position[0]];
            end
        end
        // Otherwise output zeros (during COL_ZERO_PHASE, COL_DELAY, etc.)
        
        // Column 1: Always use buffered data (has delay)
        if (col_state[1] == COL_WEIGHT_PHASE) begin
            B1 = weight_buffer[col_weight_position[1]];
        end
        // Otherwise output zeros
        
        // Column 2: Always use buffered data (has delay)
        if (col_state[2] == COL_WEIGHT_PHASE) begin
            B2 = weight_buffer[col_weight_position[2]];
        end
        // Otherwise output zeros
        
        // Column 3: Always use buffered data (has delay)
        if (col_state[3] == COL_WEIGHT_PHASE) begin
            B3 = weight_buffer[col_weight_position[3]];
        end
        // Otherwise output zeros
    end
    
    // Control signals
    always_comb begin
        // Read ROM when transitioning to MAIN_RUNNING or currently in MAIN_RUNNING state and buffer is not fully loaded
        // Start reading one cycle early to account for ROM's one cycle delay
        // Special case for layer 0: Only read ROM once (on transition to MAIN_RUNNING)
        if (current_layer_idx == 0) begin
            weight_rom_read_enable = (main_next_state == MAIN_RUNNING && main_current_state == MAIN_IDLE);
        end else begin
            weight_rom_read_enable = ((main_next_state == MAIN_RUNNING && main_current_state == MAIN_IDLE) ||
                                     (main_current_state == MAIN_RUNNING)) && 
                                   (buffer_row < ($clog2(KERNEL_SIZE))'(KERNEL_SIZE-1));
        end
        
        idle = (main_current_state == MAIN_IDLE || main_current_state == MAIN_DONE);
        weight_load_complete = (main_current_state == MAIN_DONE);
    end

endmodule 
