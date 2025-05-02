module array_output_buffer #(
  parameter int MAX_N  = 16,
  parameter int N_BITS = $clog2(MAX_N)
)(
  input  logic             clk,
  input  logic             reset,

  // 4 parallel write ports
  input  logic [3:0]             in_valid,
  input  logic [3:0][31:0]       in_output,
  input  logic [3:0][N_BITS-1:0] in_row,
  input  logic [3:0][N_BITS-1:0] in_col,

  // single read port
  output logic                   out_valid,
  output logic [31:0]            out_output,
  output logic [N_BITS-1:0]      out_row,
  output logic [N_BITS-1:0]      out_col,
  input  logic                   out_consume
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
  logic [3:0]        write_offset; // one-hot temp to compute offsets
  logic [2:0]        count;

  // Combinational: count writes, detect read, compute new pointers & count
  logic [2:0]        writes;
  logic              do_read;
  logic [2:0]        count_new;
  logic [1:0]        wr_ptr_new, rd_ptr_new;
  logic [3:0]        offset;       // for each port, number of prior valid writes

  always_comb begin
    writes    = in_valid[0] + in_valid[1] + in_valid[2] + in_valid[3];
    do_read   = out_consume && buffer[rd_ptr].valid;
    count_new = count + writes - (do_read ? 1 : 0);
    wr_ptr_new = wr_ptr + writes;
    rd_ptr_new = rd_ptr + (do_read ? 1 : 0);

    // compute write offsets per port
    offset = '0;
    int acc = 0;
    for (int i = 0; i < 4; i++) begin
      offset[i] = acc;
      if (in_valid[i]) acc++;
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
          int idx = wr_ptr + offset[i];
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
