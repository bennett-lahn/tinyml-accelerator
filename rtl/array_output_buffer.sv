`include "sys_types.svh"

module array_output_buffer #(
  parameter int MAX_N  = 16,
  parameter int N_BITS = $clog2(MAX_N)
)(
  input  logic                    clk
  ,input  logic                   reset

  // 4 parallel write ports
  ,input  logic                in_valid  [3:0] // High if input from port is valid
  ,input  int32_t              in_output [3:0] // Unquantized input value
  ,input  logic   [N_BITS-1:0] in_row    [3:0] // Row in matrix of data input
  ,input  logic   [N_BITS-1:0] in_col    [3:0] // Column in matrix of data input

  // single read port
  ,output logic                   out_valid    // High if output is valid and should be read
  ,output int32_t                 out_output   // Unquantized output value
  ,output logic [N_BITS-1:0]      out_row      // Row in matrix of data output
  ,output logic [N_BITS-1:0]      out_col      // Column in matrix of data output
  ,input  logic                   out_consume  // High if connected quantize/activate unit uses output
);

  // A single buffer entry
  typedef struct packed {
    logic [31:0]       output_val;
    logic [N_BITS-1:0] row;
    logic [N_BITS-1:0] col;
    logic              valid;
  } entry_t;

  entry_t            buffer [4];
  logic [1:0]        wr_ptr, rd_ptr;
  logic [2:0]        count;

  // Combinational: count writes, detect read, compute new pointers & count
  logic [1:0]        writes;
  logic              do_read;
  logic [2:0]        count_new;
  logic [1:0]        wr_ptr_new, rd_ptr_new;
  logic [1:0]        acc;
  logic [1:0]        offset [0:3];       // for each port, number of prior valid writes

  always_comb begin
    // If writing from all 4 ports, writes wraps back to 0, this is ok because write pointer doesn't need to move in this case
    // (I THINK). TODO: Check this in testing
    writes    = {1'b0, in_valid[0]} + {1'b0, in_valid[1]} + {1'b0, in_valid[2]} + {1'b0, in_valid[3]};
    do_read   = out_consume && buffer[rd_ptr].valid;
    count_new = count + writes - (do_read ? 1 : 0);
    wr_ptr_new = wr_ptr + writes;
    rd_ptr_new = rd_ptr + (do_read ? 1 : 0);

    // compute write offsets per port
    offset = {'b0, 'b0, 'b0, 'b0};
    acc = 'b0;
    for (int i = 0; i < 4; i++) begin
      offset[i] = acc;
      if (in_valid[i]) acc = acc + 2'd1;
    end

    // overflow check
    if (count_new > 4) begin
      $display("ERROR: Buffer overflow at time %0t! count=%0d, writes=%0d, reads=%0d", 
               $time, count, writes, do_read ? 1 : 0);
    end
  end

  // Sequential: update buffer, pointers, and count
  always_ff @(posedge clk) begin
    if (reset) begin
      count   <= 0;
      wr_ptr  <= 0;
      rd_ptr  <= 0;
      for (int i = 0; i < 4; i++)
        buffer[i].valid <= 1'b0;
    end else begin
      // perform all writes
      for (int i = 0; i < 4; i++) begin
        if (in_valid[i] && (count_new <= 4)) begin
          logic [1:0] idx = wr_ptr + offset[i];
          buffer[idx].output_val <= in_output[i];
          buffer[idx].row        <= in_row[i];
          buffer[idx].col        <= in_col[i];
          buffer[idx].valid      <= 1'b1;
        end
      end

      // perform read (invalidate oldest)
      if (do_read) begin
        buffer[rd_ptr].valid <= 1'b0;
      end

      // update pointers & count
      count  <= count_new;
      wr_ptr <= wr_ptr_new;
      rd_ptr <= rd_ptr_new;
    end
  end

  // Output logic: always show the oldest valid entry if any
  always_comb begin
    out_valid  = buffer[rd_ptr].valid;
    out_output = buffer[rd_ptr].output_val;
    out_row    = buffer[rd_ptr].row;
    out_col    = buffer[rd_ptr].col;
  end

endmodule
