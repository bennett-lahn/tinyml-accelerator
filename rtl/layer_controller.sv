`include "sys_types.svh"

// Simplified Layer Controller
// Manages: current layer, current output channel, and current tile within that output channel
// Does NOT manage: input channels or kernel locations (handled by other modules)

module layer_controller #(
  parameter int NUM_LAYERS                = 6
  ,parameter int MAX_NUM_CH				    	  = 64
  ,parameter int MAX_N                    = 512 // Max matrix dimension

  // Convolution parameters per layer
  ,parameter int CONV_IN_H   [NUM_LAYERS] = '{32,  16, 8,  4,  1,  1}
  ,parameter int CONV_IN_W   [NUM_LAYERS] = '{32,  16, 8,  4,  1,  1}
  ,parameter int CONV_OUT_H  [NUM_LAYERS] = '{16,  8,  4,  2,  1,  1}
  ,parameter int CONV_OUT_W  [NUM_LAYERS] = '{16,  8,  4,  2,  1,  1}
  ,parameter int CONV_IN_C   [NUM_LAYERS] = '{1,  8,  16,  32, 256,64}
  ,parameter int CONV_OUT_C  [NUM_LAYERS] = '{8,  16, 32,  64, 64, 10}
)(
  input  logic                     clk
  ,input  logic                    reset
  ,input  logic                    start       // Pulse to begin sequence
  ,input  logic                    stall
  ,input  logic					           sta_idle    // STA signals completion of current tile
  ,input  logic                    done        // Signals completion of current tile computation

  // Current layer and channel index
  ,output logic [$clog2(NUM_LAYERS)-1:0] layer_idx // Index of current layer
  ,output logic [$clog2(MAX_NUM_CH)-1:0] chnnl_idx // Index of current output channel / filter

  // Drive STA controller and memory conv parameters
  ,output logic                    reset_sta
  ,output logic [15:0]             mat_size
  ,output logic					           load_bias
  ,output logic                    start_compute
  ,output logic [$clog2(MAX_N)-1:0] controller_pos_row
  ,output logic [$clog2(MAX_N)-1:0] controller_pos_col
  ,output logic                     pe_mask [SA_N*SA_N] // Mask for active PEs in current tile

  // Current layer filter dimensions
  ,output logic [$clog2(MAX_NUM_CH+1)-1:0] num_filters      // Number of output channels (filters) for current layer
  ,output logic [$clog2(MAX_NUM_CH+1)-1:0] num_input_channels // Number of input channels for current layer

  // Drive STA controller pool parameters
  ,output logic 				           bypass_maxpool
);

  // Systolic array height/width
  localparam SA_N = 4;

  // State encoding
  typedef enum logic [2:0] {
    S_IDLE,
    S_RESET_STA,
    S_LOAD_BIAS_1,
    S_LOAD_BIAS_2,
    S_RUN,
    S_DONE
  } state_t;

  state_t current_state, next_state;
  
  // Main counters: layer, channel, and tile
  logic [$clog2(NUM_LAYERS+1)-1:0] layer_count;
  logic [$clog2(MAX_NUM_CH+1)-1:0] channel_count;
  logic [$clog2(64)-1:0] current_tile_row, current_tile_col;

  // Tiling control
  logic [$clog2(64)-1:0] tiles_per_row, tiles_per_col;
  logic [15:0] current_out_h, current_out_w;
  logic last_tile, last_channel, last_layer;

  // Tile dimension calculations
  logic [$clog2(SA_N+1)-1:0] current_tile_h, current_tile_w;
  logic [$clog2(SA_N+1)-1:0] remaining_out_h, remaining_out_w;

  // State register and counter logic
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      current_state    <= S_IDLE;
      layer_count      <= 'd0;
      channel_count    <= 'd0;
      current_tile_row <= 'd0;
      current_tile_col <= 'd0;
    end else begin
      current_state <= next_state;
      
      // Advance counters when transitioning from S_RUN to S_RESET_STA (tile completion)
      if (current_state == S_RUN && next_state == S_RESET_STA && sta_idle && done) begin
        // Advance tile position
        if (current_tile_col == tiles_per_col - 1) begin
          current_tile_col <= 'd0;
          if (current_tile_row == tiles_per_row - 1) begin
            current_tile_row <= 'd0;
            // Advance to next channel
            if (channel_count == CONV_OUT_C[layer_count] - 1) begin
              channel_count <= 'd0;
              // Advance to next layer
              layer_count <= layer_count + 1'd1;
            end else begin
              channel_count <= channel_count + 1'd1;
            end
          end else begin
            current_tile_row <= current_tile_row + 1'd1;
          end
        end else begin
          current_tile_col <= current_tile_col + 1'd1;
        end
      end
    end
  end

  // Calculate tiling parameters for current layer
  always_comb begin
    current_out_h = CONV_OUT_H[layer_count];
    current_out_w = CONV_OUT_W[layer_count];
    
    // Calculate number of 4x4 tiles needed using ceil(current_out_[h/w]/SA_N)
    tiles_per_row = (current_out_h + SA_N - 1) / SA_N;
    tiles_per_col = (current_out_w + SA_N - 1) / SA_N;
    
    // Check if this is the last tile/channel/layer
    last_tile    = (current_tile_row == tiles_per_row - 1) && 
                   (current_tile_col == tiles_per_col - 1);
    last_channel = (channel_count == CONV_OUT_C[layer_count] - 1);
    last_layer   = (layer_count == NUM_LAYERS - 1);

    // Calculate remaining output dimensions from current tile position
    remaining_out_h = current_out_h - (current_tile_row * SA_N);
    remaining_out_w = current_out_w - (current_tile_col * SA_N);
    
    // Current tile dimensions are min(SA_N, remaining_dimensions)
    current_tile_h = (remaining_out_h < SA_N) ? remaining_out_h : SA_N;
    current_tile_w = (remaining_out_w < SA_N) ? remaining_out_w : SA_N;

    // Generate PE mask for current tile
    for (int i = 0; i < SA_N; i++) begin
      for (int j = 0; j < SA_N; j++) begin
        int flat_idx = i * SA_N + j;
        // PE is active if within current tile bounds
        pe_mask[flat_idx] = (i < current_tile_h) && (j < current_tile_w);
      end
    end
  end

  // Next‐state logic and control signals
  always_comb begin
    // Defaults
    next_state       = current_state;
    load_bias        = 1'b0;
    start_compute    = 1'b0;
    reset_sta        = 1'b0;

    // Output current layer parameters
    mat_size         = (current_out_h > current_out_w) ? current_out_h : current_out_w;
    bypass_maxpool   = (layer_count == NUM_LAYERS - 1); // Last layer bypasses pool

    // Calculate current tile position  
    controller_pos_row = current_tile_row * SA_N;
    controller_pos_col = current_tile_col * SA_N;
    
    // Output current indices
    layer_idx = layer_count;
    chnnl_idx = channel_count;

    // Output current layer filter dimensions
    num_filters        = CONV_OUT_C[layer_count];  
    num_input_channels = CONV_IN_C[layer_count];

    case (current_state)
      S_IDLE: begin
        if (start)
          next_state = S_RESET_STA;
      end

      S_RESET_STA: begin
        reset_sta = 1'b1;
        next_state = S_LOAD_BIAS_1;
      end

      S_LOAD_BIAS_1: begin
        reset_sta   = 1'b0;
        load_bias   = 1'b1;
        next_state  = S_LOAD_BIAS_2;
      end

      S_LOAD_BIAS_2: begin
        load_bias  = 1'b0;
        next_state = S_RUN;
      end

      S_RUN: begin
        start_compute = 1'b1;
        
        if (stall) begin
          next_state = S_RUN;
          start_compute = 1'b0;
        end else if (sta_idle && done) begin 
          // STA has completed the current tile computation and done is asserted
          start_compute = 1'b0;
          
          if (last_tile && last_channel && last_layer) begin
            next_state = S_DONE;
          end else begin
            // Move to next tile/channel/layer
            next_state = S_RESET_STA;
          end
        end else begin 
          // STA is still computing current tile or done is not yet asserted
          next_state = S_RUN;
        end
      end

      S_DONE: begin
        next_state = S_IDLE;
      end
    endcase
  end

endmodule

// Each PE has a VECTOR_WIDTH = 4. Does this mean each PE could take inputs from 4 different channels at a time?

// TODO:
// 1. start_compute does not do what it needs to 
// 2. STA idle should only be expected to trigger when the output is completely finished calculating (i.e., at the end of that output tile, but not a finer granularity).

// output coordinator calculation for compute time needs to be updated
