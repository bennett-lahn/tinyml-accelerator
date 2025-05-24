`include "sys_types.svh"

// TODO: Currently, this module waits until the entire pipeline is empty before beginning
// calculations of the next layer/channel, but this should be made more efficient in the future

// TODO: Layer controller currently inaccurately calculates the output for each channel because
// it does not properly tile across input channels
module layer_controller #(
  parameter int NUM_LAYERS                = 6
  ,parameter int MAX_NUM_CH				    	  = 64
  ,parameter int MAX_NUM_CH               = 512 // Max matrix dimension

  // Convolution parameters per layer
  ,parameter int CONV_IN_H   [NUM_LAYERS] = '{32,  16, 8,  4,  1,  1}
  ,parameter int CONV_IN_W   [NUM_LAYERS] = '{32,  16, 8,  4,  1,  1}
  ,parameter int CONV_OUT_H  [NUM_LAYERS] = '{16,  8,  4,  2,  1,  1}
  ,parameter int CONV_IN_H   [NUM_LAYERS] = '{16,  8,  4,  2,  1,  1}
  ,parameter int CONV_IN_C   [NUM_LAYERS] = '{1,  8,  16,  32, 256,64}
  ,parameter int CONV_OUT_C  [NUM_LAYERS] = '{8,  16, 32,  64, 64, 10}
  ,parameter int CONV_KH     [NUM_LAYERS] = '{4,  4,   4,  4,  1,  1}
  ,parameter int CONV_KW     [NUM_LAYERS] = '{4,  4,   4,  4,  1,  1}
  ,parameter int CONV_STR_H  [NUM_LAYERS] = '{1,  1,   1,  1,  1,  1}
  ,parameter int CONV_STR_W  [NUM_LAYERS] = '{1,  1,   1,  1,  1,  1}
)(
  input  logic                     clk
  ,input  logic                    reset
  ,input  logic                    start       // pulse to begin sequence
  ,input  logic                    stall
  ,input  logic					           sta_idle

  ,output logic                    busy        // high while stepping layers
  ,output logic                    done        // pulse when all layers finished

  // Current layer and channel index
  ,output logic [$clog2(NUM_LAYERS)-1:0]   layer_idx
  ,output logic [$clog2(MAX_NUM_CH+1)-1:0] chnnl_idx

  // Drive STA controller and memory conv parameters
  ,output logic                    reset_sta
  ,output logic [15:0]             mat_size
  ,output logic					           load_bias
  ,output logic                    start_compute
  ,output logic [$clog2(MAX_NUM_CH+1)-1:0] controller_pos_row
  ,output logic [$clog2(MAX_NUM_CH+1)-1:0] controller_pos_col
  ,output logic                   pe_mask [SA_N*SA_N] // Mask for active PEs in current tile
  ,output logic [7:0]             kernel_h
  ,output logic [7:0]             kernel_w
  ,output logic [7:0]             stride_h
  ,output logic [7:0]             stride_w

  // Drive STA controller pool parameters
  ,output logic 				           bypass_maxpool
);

  // Systolic array height/width
  localparam SA_N = 4;

  // State encoding
  typedef enum logic [1:0] {
    S_IDLE
    ,S_RESET_STA
    ,S_LOAD_BIAS_1
    ,S_LOAD_BIAS_2
    ,S_RUN
    ,S_DONE
  } state_t;

  state_t current_state, next_state;
  logic [$clog2(NUM_LAYERS+1)-1:0] layer_count;
  logic [$clog2(MAX_NUM_CH+1)-1:0] channel_count;

  // Tiling control
  logic [$clog2(64)-1:0] tiles_per_row, tiles_per_col, total_tiles;
  logic [$clog2(64)-1:0] current_tile_row, current_tile_col, current_tile_idx;
  logic [15:0] current_out_h, current_out_w;
  logic last_tile, last_channel, last_layer;
  // Tile dimension calculations
  logic [$clog2(SA_N+1)-1:0] current_tile_h, current_tile_w;
  logic [$clog2(SA_N+1)-1:0] remaining_out_h, remaining_out_w;

  // State register
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      current_state   <= S_IDLE;
      layer_count     <= 'd0;
      channel_count   <= 'd0;
      current_tile_row <= 'd0;
      current_tile_col <= 'd0;
      current_tile_idx <= 'd0;
    end else begin
      current_state <= next_state;
      
      // Advance counters based on state transitions
      if (current_state == S_RUN && next_state == S_RESET_STA && sta_idle) begin
        // Move to next tile/channel/layer
        if (current_tile_col == tiles_per_col - 1) begin
          current_tile_col <= 'd0;
          if (current_tile_row == tiles_per_row - 1) begin
            current_tile_row <= 'd0;
            // Move to next channel
            if (channel_count == CONV_OUT_C[layer_count] - 1) begin
              channel_count <= 'd0;
              layer_count   <= layer_count + 'd1;
            end else begin
              channel_count <= channel_count + 'd1;
            end
          end else begin
            current_tile_row <= current_tile_row + 'd1;
          end
        end else begin
          current_tile_col <= current_tile_col + 'd1;
        end
        current_tile_idx <= current_tile_idx + 'd1;
      end else if (current_state == S_LOAD_BIAS_2 && next_state == S_RUN) begin
        // Reset tile counters for new computation
        current_tile_row <= 'd0;
        current_tile_col <= 'd0;
        current_tile_idx <= 'd0;
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
    total_tiles   = tiles_per_row * tiles_per_col;
    
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
    busy             = 1'b0;
    done             = 1'b0;
    load_bias        = 1'b0;
    start_compute    = 1'b0;
    reset_sta        = 1'b0;

    // Output current layer parameters
    mat_size         = (current_out_h > current_out_w) ? current_out_h : current_out_w;
    kernel_h         = CONV_KH[layer_count];
    kernel_w         = CONV_KW[layer_count];
    stride_h         = CONV_STR_H[layer_count];
    stride_w         = CONV_STR_W[layer_count];
    bypass_maxpool   = (layer_count == NUM_LAYERS - 1); // Last layer bypasses pool

    // Calculate current tile position  
    controller_pos_row = current_tile_row * SA_N;
    controller_pos_col = current_tile_col * SA_N;
    
    layer_idx        = layer_count;
    chnnl_idx        = channel_count;

    case (current_state)
      S_IDLE: begin
        if (start)
          next_state = S_RESET_STA;
      end

      S_RESET_STA: begin
        busy      = 1'b1;
        reset_sta = 1'b1;
        next_state = S_LOAD_BIAS_1;
      end

      S_LOAD_BIAS_1: begin
        busy        = 1'b1;
        reset_sta   = 1'b0;
        load_bias   = 1'b1;
        next_state  = S_LOAD_BIAS_2;
      end

      S_LOAD_BIAS_2: begin
        busy       = 1'b1;
        load_bias  = 1'b0;
        next_state = S_RUN;
      end

      S_RUN: begin
        busy          = 1'b1;
        start_compute = 1'b1;
        
        if (stall) begin
          next_state = S_RUN;
        end else if (sta_idle) begin
          // Move to next tile/channel/layer when STA completes
          if (last_tile && last_channel && last_layer) begin
            next_state = S_DONE;
          end else if (last_tile && last_channel) begin
            next_state = S_RESET_STA; // New layer, reset STA then reload bias
          end else if (last_tile) begin
            next_state = S_RESET_STA; // New channel, reset STA then reload bias  
          end else begin
            next_state = S_RESET_STA; // New tile, reset STA then reload bias
          end
        end else begin
          next_state = S_RUN;
        end
      end

      S_DONE: begin
        done       = 1'b1;
        next_state = S_IDLE;
      end
    endcase
  end

endmodule
