`include "sys_types.svh"

// TODO: Consider how buffer bypassing affects critical path, whether it is necessary

module array_output_buffer #(
  parameter int MAX_N  = 512
  ,parameter int N_BITS = $clog2(MAX_N)
  ,parameter int NUM_WRITE_PORTS = 4
  ,parameter int PTR_BITS = $clog2(MAX_BUFFER_ENTRIES) // For wr_ptr, rd_ptr (0 to 3) -> 2 bits
  ,parameter int MAX_BUFFER_ENTRIES = 4
  ,parameter int COUNT_BITS = $clog2(MAX_BUFFER_ENTRIES + 1) // For count (0 to 4) -> 3 bits
)(
  input  logic               clk
  ,input  logic              reset

  // 4 parallel write ports
  ,input  logic              in_valid  [NUM_WRITE_PORTS] // High if input from port is valid
  ,input  int32_t            in_output [NUM_WRITE_PORTS] // Unquantized input value
  ,input  logic [N_BITS-1:0] in_row    [NUM_WRITE_PORTS] // Row in matrix of data input
  ,input  logic [N_BITS-1:0] in_col    [NUM_WRITE_PORTS] // Column in matrix of data input

  // single read port
  ,output logic              idle            // High if buffer is empty and has no incoming data
  ,output logic              out_valid       // High if output is valid and should be read
  ,output int32_t            out_output      // Unquantized output value
  ,output logic [N_BITS-1:0] out_row         // Row in matrix of data output
  ,output logic [N_BITS-1:0] out_col         // Column in matrix of data output
  ,input  logic              out_consume     // High if connected quantize/activate unit uses output
);

  // A single buffer entry
  typedef struct packed {
    int32_t            output_val;
    logic [N_BITS-1:0] row;
    logic [N_BITS-1:0] col;
    logic              valid;
  } entry_t;

  // Buffer memory and state registers
  entry_t                buffer [MAX_BUFFER_ENTRIES];
  logic [PTR_BITS-1:0]   wr_ptr, rd_ptr;
  logic [COUNT_BITS-1:0] count;

  // Combinational logic signals
  logic is_bypassing;                              // True if an input is bypassing the buffer
  logic [PTR_BITS-1:0] bypass_input_idx;           // Index of the input chosen for bypass
  int32_t bypass_data_output;                      // Data for bypass output
  logic [N_BITS-1:0] bypass_data_row;              // Row for bypass output
  logic [N_BITS-1:0] bypass_data_col;              // Col for bypass output

  logic [COUNT_BITS-1:0] writes_total_count;       // How many in_valid signals are high
  logic actual_read_from_buffer;                   // True if consuming an entry from the buffer memory
  logic [COUNT_BITS-1:0] writes_to_buffer_count;   // How many write operations will target the buffer memory

  logic [COUNT_BITS-1:0] count_next_proposed;      // Proposed value for count for the next cycle (before saturation)
  logic [PTR_BITS-1:0]   wr_ptr_next, rd_ptr_next; // Proposed values for pointers

  logic signed [COUNT_BITS:0] temp_next_count;     // Calculates new buffer count, detect over/underflow
  logic [PTR_BITS-1:0] curr_buffer_write_offset;   // Assigns written values to correct index offset

  // For each input port, its target slot offset relative to wr_ptr if it's written to the buffer
  logic [PTR_BITS-1:0]   write_slot_offset [0:NUM_WRITE_PORTS-1];

  always_comb begin
    // Initialize default values for combinational signals
    writes_total_count = '0;
    is_bypassing = 1'b0;
    bypass_input_idx = 'x; 
    bypass_data_output = 'x;
    bypass_data_row = 'x;
    bypass_data_col = 'x;
    actual_read_from_buffer = 1'b0;
    writes_to_buffer_count = '0;
    count_next_proposed = 'x;
    wr_ptr_next = wr_ptr;
    rd_ptr_next = rd_ptr;
    for (int i=0; i<NUM_WRITE_PORTS; i++) write_slot_offset[i] = '0;

    // 1. Calculate total number of incoming valid writes
    for (int i = 0; i < NUM_WRITE_PORTS; i++) begin
      if (in_valid[i]) begin
        writes_total_count = writes_total_count + 1;
      end
    end

    // 2. Determine bypass logic:
    // Bypass occurs if: consumer wants data (out_consume), buffer is empty (count == 0),
    // AND there is at least one valid input (writes_total_count > 0).
    if (out_consume && (count == 0) && (writes_total_count > 0)) begin
      is_bypassing = 1'b1;
      // Select the first valid input port for bypass (priority: port 0 > 1 > 2 > 3)
      for (int i = 0; i < NUM_WRITE_PORTS; i++) begin
        if (in_valid[i]) begin
          bypass_input_idx   = i[PTR_BITS-1:0]; 
          bypass_data_output = in_output[i];
          bypass_data_row    = in_row[i];
          bypass_data_col    = in_col[i];
          break;
        end
      end
    end

    // 3. Determine if an actual read from the buffer memory will occur
    actual_read_from_buffer = out_consume & buffer[rd_ptr].valid & ~is_bypassing;

    // 4. Calculate how many writes will actually go into the buffer memory
    if (is_bypassing) begin
      writes_to_buffer_count = (writes_total_count > 0) ? (writes_total_count - 1) : COUNT_BITS'(0);
    end else begin
      writes_to_buffer_count = writes_total_count;
    end
    
    // 5. Calculate proposed next state for count, write pointer, and read pointer
    count_next_proposed = count + writes_to_buffer_count - {{(COUNT_BITS-1){1'b0}}, actual_read_from_buffer};
    
    wr_ptr_next = PTR_BITS'(wr_ptr + writes_to_buffer_count); // Pointer advances by # items buffered; wraps
    rd_ptr_next = rd_ptr + actual_read_from_buffer; // Pointer advances if read; wraps

    // 6. Compute write slot offsets for inputs that are actually written to the buffer
    curr_buffer_write_offset = 0;
    for (int i = 0; i < NUM_WRITE_PORTS; i++) begin
      if (in_valid[i]) begin
        if ((is_bypassing && !(i[1:0] == bypass_input_idx)) || !is_bypassing) begin // Ignore bypassed inputs
          write_slot_offset[i] = curr_buffer_write_offset;
          curr_buffer_write_offset = curr_buffer_write_offset + 1;
        end
      end
    end

    // 7. Calculate next count for buffer (use larger vector to prevent over/underflow)
    temp_next_count = count + writes_to_buffer_count;
    if (actual_read_from_buffer) begin
        temp_next_count = temp_next_count - 1;
    end
  end  // always_comb

  always_ff @(posedge clk) begin
    if (reset) begin
      count  <= '0;
      wr_ptr <= '0;
      rd_ptr <= '0;
      for (int i = 0; i < MAX_BUFFER_ENTRIES; i++) begin
        buffer[i].valid <= 1'b0;
      end
    end else begin
      // Perform writes to the buffer for non-bypassed inputs
      for (int i = 0; i < NUM_WRITE_PORTS; i++) begin
        if (in_valid[i]) begin
          // Check if this input is being bypassed
          if (!(is_bypassing && (PTR_BITS'(i) == bypass_input_idx))) begin
            // This input is NOT bypassed, so write it to the buffer.
            buffer[wr_ptr + write_slot_offset[i]].output_val <= in_output[i];
            buffer[wr_ptr + write_slot_offset[i]].row        <= in_row[i];
            buffer[wr_ptr + write_slot_offset[i]].col        <= in_col[i];
            buffer[wr_ptr + write_slot_offset[i]].valid      <= 1'b1;
          end
        end
      end

      // Perform read (invalidate) if an actual read happened
      if (actual_read_from_buffer) begin
        buffer[rd_ptr].valid <= 1'b0;
      end
      
      // Prevent underflow or overflow
      if (temp_next_count > MAX_BUFFER_ENTRIES[COUNT_BITS:0]) begin
        count <= COUNT_BITS'(MAX_BUFFER_ENTRIES);
      end else if (temp_next_count < 0) begin
        count <= COUNT_BITS'(0); // Should not happen if actual_read_from_buffer implies count > 0
      end else begin
        count <= temp_next_count[COUNT_BITS-1:0];
      end

      if (temp_next_count > MAX_BUFFER_ENTRIES[COUNT_BITS:0]) begin
      if (!reset) // Avoid messages during reset
        $display("ERROR: Buffer overflow condition at time %0t! current_count=%d, writes_to_buffer=%d, read_from_buffer=%b, proposed_raw_next_count=%d",
                 $time, count, writes_to_buffer_count, actual_read_from_buffer, temp_next_count);
    end
    if (temp_next_count < 0) begin
       if (!reset) // Avoid messages during reset
        $display("WARNING: Buffer underflow condition at time %0t! current_count=%d, writes_to_buffer=%d, read_from_buffer=%b, proposed_raw_next_count=%d",
                 $time, count, writes_to_buffer_count, actual_read_from_buffer, temp_next_count);
    end
      
      wr_ptr <= wr_ptr_next;
      rd_ptr <= rd_ptr_next;
    end
  end // always_ff

  // Debug display
  // always_ff @(posedge clk) begin
  //   if (!reset) begin
  //       // Display current state and key combinational decisions that led to the *next* state update
  //       // This provides a snapshot at the end of the cycle, reflecting inputs and decisions for that cycle.
  //       $display("Time: %0t, Cycle End State: count=%d, wr_ptr=%h, rd_ptr=%h. Inputs: out_consume=%b",
  //                $time, count, wr_ptr, rd_ptr, out_consume);
  //       for (int i=0; i<NUM_WRITE_PORTS; i++) begin
  //           if(in_valid[i]) $display("  InPort[%d]: valid, data_val=%x, row=%d, col=%d", i, in_output[i], in_row[i], in_col[i]);
  //       end
  //       $display("  Comb Decisions: is_bypassing=%b, bypass_idx=%h (if bypassing), writes_total=%d, writes_to_buffer=%d, read_from_buffer=%b",
  //                is_bypassing, bypass_input_idx, writes_total_count, writes_to_buffer_count, actual_read_from_buffer);
  //       $display("  Comb Next Ptrs/Count (proposed): count_next_prop=%d, wr_ptr_next=%h, rd_ptr_next=%h",
  //                count_next_proposed, wr_ptr_next, rd_ptr_next);
  //   end
  // end // always_ff

  // Output MUX: Selects bypass data or buffer data
  always_comb begin
    if (is_bypassing) begin
      out_valid  = 1'b1;
      out_output = bypass_data_output;
      out_row    = bypass_data_row;
      out_col    = bypass_data_col;
    end else begin
      // Standard output from buffer's read pointer
      out_valid  = buffer[rd_ptr].valid; // Implies count > 0 if this valid is high
      out_output = buffer[rd_ptr].output_val;
      out_row    = buffer[rd_ptr].row;
      out_col    = buffer[rd_ptr].col;
    end
    idle = (count == '0 && in_valid == '0) ? 1'b1 : 1'b0; 
  end // always_comb

endmodule
