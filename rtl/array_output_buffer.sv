`include "sys_types.svh"

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

  // Simple buffer array - 4 entries max
  typedef struct packed {
    int32_t            output_val;
    logic [N_BITS-1:0] row;
    logic [N_BITS-1:0] col;
    logic              valid;
  } entry_t;

  entry_t buffer [MAX_BUFFER_ENTRIES];
  logic [COUNT_BITS-1:0] count;
  logic [PTR_BITS-1:0] rd_ptr;

  // Count valid inputs this cycle
  logic [COUNT_BITS-1:0] valid_inputs_count;
  
  always_comb begin
    valid_inputs_count = '0;
    for (int i = 0; i < NUM_WRITE_PORTS; i++) begin
      if (in_valid[i]) begin
        valid_inputs_count = valid_inputs_count + 1;
      end
    end
  end

  // Write logic: store incoming valid data
  always_ff @(posedge clk) begin
    if (reset) begin
      count <= '0;
      rd_ptr <= '0;
      for (int i = 0; i < MAX_BUFFER_ENTRIES; i++) begin
        buffer[i].valid <= 1'b0;
      end
    end else begin
      // Write new data if there's space and valid inputs
      if (valid_inputs_count > 0 && (count + valid_inputs_count <= MAX_BUFFER_ENTRIES)) begin
        int write_idx = count;
        for (int i = 0; i < NUM_WRITE_PORTS; i++) begin
          if (in_valid[i] && write_idx < MAX_BUFFER_ENTRIES) begin
            buffer[write_idx].output_val <= in_output[i];
            buffer[write_idx].row        <= in_row[i];
            buffer[write_idx].col        <= in_col[i];
            buffer[write_idx].valid      <= 1'b1;
            write_idx++;
          end
        end
      end

      // Read/consume logic
      if (out_consume && buffer[rd_ptr].valid) begin
        buffer[rd_ptr].valid <= 1'b0;
        rd_ptr <= (rd_ptr + 1) % MAX_BUFFER_ENTRIES;
        count <= count - 1;
      end

      // Update count for writes
      // This logic is only acceptable because reads and writes should never overlap
      if (valid_inputs_count > 0 && (count + valid_inputs_count <= MAX_BUFFER_ENTRIES)) begin
        count <= count + valid_inputs_count;
      end
    end
  end

  // Output logic
  always_comb begin
    out_valid  = buffer[rd_ptr].valid;
    out_output = buffer[rd_ptr].output_val;
    out_row    = buffer[rd_ptr].row;
    out_col    = buffer[rd_ptr].col;
    
    // Idle when buffer is empty and no valid inputs
    idle = (count == '0) && (valid_inputs_count == '0);
  end

endmodule
