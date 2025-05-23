`include "sys_types.svh"

module maxpool_unit #(
  parameter  int SA_N          = 4               // Scratch array dimension; same as columns of STA, # of requant units
  ,parameter int MAX_N         = 512             // Max matrix dimension calculated
  ,parameter int N_BITS        = $clog2(MAX_N+1) // Bits to hold mat_size & coords
  ,parameter int FILTER_H      = 2               // Pooling window height
  ,parameter int FILTER_W      = 2               // Pooling window width
) (
  input  logic                clk
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
  ,output logic                idle
  ,output logic                out_valid
  ,output logic [N_BITS-1:0]   out_row
  ,output logic [N_BITS-1:0]   out_col
  ,output int8_t               out_data
);

  // Scratch storage and valid flags
  int8_t  scratch   [SA_N][SA_N];
  logic   valid_map [SA_N][SA_N];
  // Vector size needed to index into SA_NxSA_N signals
  localparam ARR_BITS = $clog2(SA_N);
  // Vector size needed to index blocks after pooling is completed
  localparam FILTER_H_BITS = $clog2(SA_N/FILTER_H);
  localparam FILTER_W_BITS = $clog2(SA_N/FILTER_W);

  // Block-scan state
  // Used to find ready blocks and execute pooling when a block is ready
  logic                     block_ready;
  logic                     all_ok;
  logic [FILTER_H_BITS-1:0] blk_r;
  logic [FILTER_W_BITS-1:0] blk_c;
  int8_t                    maxv;
  logic [FILTER_H_BITS-1:0] off_r;
  logic [FILTER_W_BITS-1:0] off_c;

  logic [N_BITS-1:0]        comb_out_row;
  logic [N_BITS-1:0]        comb_out_col;
  int8_t                    comb_out_data;

  // Read incoming samples into scratch
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      for (int r = 0; r < SA_N; r++) begin
        for (int c = 0; c < SA_N; c++)
          valid_map[r][c] <= 1'b0;
      end
      out_valid <= 1'b0;
      out_row   <= '0;
      out_col   <= '0;
      out_data  <= '0;
    end else begin
      // Write new samples
      for (int i = 0; i < SA_N; i++) begin
        if (in_valid[i]) begin
          scratch[ARR_BITS'(in_row[i]-pos_row)][ARR_BITS'(in_col[i]-pos_col)]   <= in_data[i];
          valid_map[ARR_BITS'(in_row[i]-pos_row)][ARR_BITS'(in_col[i]-pos_col)] <= 1'b1;
        end
      end
      // Clear the 2×2 (or H×W) block about to emit
      if (block_ready) begin
        for (int pr = 0; pr < FILTER_H; pr++) begin
          for (int pc = 0; pc < FILTER_W; pc++)
            valid_map[FILTER_H*blk_r+pr][FILTER_W*blk_c+pc] <= 1'b0;
        end
        out_valid <= 1'b1;
        out_row   <= comb_out_row;
        out_col   <= comb_out_col;
        out_data  <= comb_out_data;
      end else begin
        out_valid <= 1'b0;
      end
    end
  end

  // Detect first ready H×W block, find max, compute coords
  always_comb begin
    idle           = ~out_valid;
    block_ready    = 1'b0;
    off_r          = 1'b0;
    off_c          = 1'b0;
    all_ok         = 1'b1;
    blk_r          = '0;
    blk_c          = '0;
    comb_out_data  = '0;
    comb_out_row   = '0;
    comb_out_col   = '0;
    maxv           = '0;

    // Scan tile rows/cols
    for (int r = 0; r < SA_N/FILTER_H; r++) begin : for1
      for (int c = 0; c < SA_N/FILTER_W; c++) begin : for2
        if (!block_ready) begin
          all_ok = 1'b1;
          for (int pr = 0; pr < FILTER_H; pr++) begin
            for (int pc = 0; pc < FILTER_W; pc++)
              all_ok &= valid_map[FILTER_H*r+pr][FILTER_W*c+pc];
          end
          if (all_ok) begin
            block_ready = 1'b1;
            blk_r = FILTER_H_BITS'(r);
            blk_c = FILTER_W_BITS'(c);
          end
        end
      end
    end

    if (block_ready) begin : blk_rdy
      // Initialize
      maxv = scratch[FILTER_H*blk_r][FILTER_W*blk_c];
      // Find max over H×W
      for (int pr = 0; pr < FILTER_H; pr++) begin : for3
        for (int pc = 0; pc < FILTER_W; pc++) begin : for4
          if (scratch[FILTER_H*blk_r+pr][FILTER_W*blk_c+pc] > maxv) begin
            maxv = scratch[FILTER_H*blk_r+pr][FILTER_W*blk_c+pc];
          end
        end
      end
      comb_out_data  = maxv;
      comb_out_row   = N_BITS'(pos_row + FILTER_H*blk_r); // (0,0) of block becomes new row/col value
      comb_out_col   = N_BITS'(pos_col + FILTER_W*blk_c);
    end

    // Idle is high if all values of valid_map are zero and out_valid is zero
    for (int i = 0; i < SA_N; i++)
      for (int j = 0; j < SA_N; j++)
        idle &= ~valid_map[i][j];
  end

endmodule
