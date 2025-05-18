`include "sys_types.svh"

module maxpool_unit #(
  ,parameter int SA_N  = 4                       // scratch array dimension; same as columns of STA, # of requant units
  ,parameter int MAX_N         = 512             // Max matrix dimension calculated
  ,parameter int N_BITS        = $clog2(MAX_N+1) // Bits to hold mat_size & coords
  ,parameter int FILTER_H      = 2               // Pooling window height
  ,parameter int FILTER_W      = 2               // Pooling window width
) (
  ,input  logic                clk
  ,input  logic                reset

  // Base coordinate for this pooling tile
  ,input  logic [N_BITS-1:0]   pos_row
  ,input  logic [N_BITS-1:0]   pos_col

  // Inputs: one per systolic array column
  ,input  logic                in_valid [SA_N]
  ,input  logic [N_BITS-1:0]   in_row   [SA_N]
  ,input  logic [N_BITS-1:0]   in_col   [SA_N]
  ,input  int8_t               in_data  [SA_N]

  // MaxPool output
  ,output logic                out_valid
  ,output logic [N_BITS-1:0]   out_row
  ,output logic [N_BITS-1:0]   out_col
  ,output int8_t               out_data
);

  // Scratch storage and valid flags
  int8_t  scratch   [SA_N][SA_N];
  logic   valid_map [SA_N][SA_N];

  // Block-scan state
  logic                             block_ready;
  logic [$clog2(SA_N/FILTER_H)-1:0] blk_r;
  logic [$clog2(SA_N/FILTER_W)-1:0] blk_c;
  int8_t                            maxv;
  logic [$clog2(FILTER_H)-1:0]      off_r;
  logic [$clog2(FILTER_W)-1:0]      off_c;

  // Read incoming samples into scratch
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      for (int r = 0; r < SA_N; r++)
        for (int c = 0; c < SA_N; c++)
          valid_map[r][c] <= 1'b0;
    end else begin
      // write new samples
      for (int ch = 0; ch < SA_N; ch++) begin
        if (in_valid[ch]) begin
          scratch[lr][lc]   <= in_data[ch];
          valid_map[lr][lc] <= 1'b1;
        end
      end
      // clear the 2×2 (or H×W) block about to emit
      if (block_ready) begin
        for (int pr = 0; pr < FILTER_H; pr++)
          for (int pc = 0; pc < FILTER_W; pc++)
            valid_map[FILTER_H*blk_r+pr][FILTER_W*blk_c+pc] <= 1'b0;
      end
    end
  end

  // Detect first ready H×W block, find max, compute coords
  always_comb begin
    block_ready = 1'b0;
    blk_r       = '0;
    blk_c       = '0;
    // scan tile rows/cols
    for (int r = 0; r < SA_N/FILTER_H; r++) begin
      for (int c = 0; c < SA_N/FILTER_W; c++) begin
        if (!block_ready) begin
          logic all_ok = 1;
          for (int pr = 0; pr < FILTER_H; pr++)
            for (int pc = 0; pc < FILTER_W; pc++)
              all_ok &= valid_map[FILTER_H*r+pr][FILTER_W*c+pc];
          if (all_ok) begin
            block_ready = 1'b1;
            blk_r = r;
            blk_c = c;
          end
        end
      end
    end

    if (block_ready) begin
      // initialize
      maxv = scratch[FILTER_H*blk_r][FILTER_W*blk_c];
      off_r = 0; off_c = 0;
      // find max over H×W
      for (int pr = 0; pr < FILTER_H; pr++) begin
        for (int pc = 0; pc < FILTER_W; pc++) begin
          int8_t v = scratch[FILTER_H*blk_r+pr][FILTER_W*blk_c+pc];
          if (v > maxv) begin
            maxv  = v;
          end
        end
      end
      out_data  = maxv;
      out_row   = pos_row + FILTER_H*blk_r; // (0,0) of block becomes new row/col value
      out_col   = pos_col + FILTER_W*blk_c;
      out_valid = 1'b1;
    end else begin
      out_data  = '0;
      out_row   = '0;
      out_col   = '0;
      out_valid = 1'b0;
    end
  end

endmodule
