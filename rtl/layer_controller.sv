`include "sys_types.svh"

module layer_controller #(
  parameter int NUM_LAYERS = 4
  ,parameter int NUM_CH					  = 64

  // Convolution parameters per layer
  ,parameter int CONV_IN_H   [NUM_LAYERS] = '{32, 16,  8,  4}
  ,parameter int CONV_IN_W   [NUM_LAYERS] = '{32, 16,  8,  4}
  ,parameter int CONV_IN_C   [NUM_LAYERS] = '{1,  8,  16,  32}
  ,parameter int CONV_OUT_C  [NUM_LAYERS] = '{8,  16, 32,  64}
  ,parameter int CONV_KH     [NUM_LAYERS] = '{4,  4,   4,  4}
  ,parameter int CONV_KW     [NUM_LAYERS] = '{4,  4,   4,  4}
  ,parameter int CONV_STR_H  [NUM_LAYERS] = '{1,  1,   1,  1}
  ,parameter int CONV_STR_W  [NUM_LAYERS] = '{1,  1,   1,  1}
)(
  input  logic                     clk
  ,input  logic                    reset
  ,input  logic                    start       // pulse to begin sequence
  ,input  logic					   sta_idle
  ,output logic                    busy        // high while stepping layers
  ,output logic                    done        // pulse when all layers finished

  // Current layer index
  ,output logic [$clog2(NUM_LAYERS)-1:0] layer_idx

  // Drive STA controller and memory conv parameters
  ,output logic [15:0]             mat_size
  ,output logic					   load_bias
  ,output logic [15:0]             in_ch
  ,output logic [15:0]             out_ch
  ,output logic [15:0]             kernel_h
  ,output logic [15:0]             kernel_w
  ,output logic [15:0]             stride_h
  ,output logic [15:0]             stride_w

  // Drive STA controller pool parameters
  ,output logic 				   bypass_maxpool
);

  // State encoding
  typedef enum logic [1:0] {
    S_IDLE
    ,S_LOAD_BIAS_1
    ,S_LOAD_BIAS_2
    ,S_RUN
    ,S_STALL
    ,S_WAIT
    ,S_DONE
  } state_t;

  state_t current_state, next_state;
  logic [$clog2(NUM_LAYERS)-1:0] layer_count;

  // State register
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      current_state <= S_IDLE;
      layer_count   <= '0;
    end else begin
      current_state <= next_state;
      if (current_state == S_LOAD && next_state == S_RUN)
        layer_count <= layer_count + 1'b1;
    end
  end

  // Next‐state logic and control signals
  always_comb begin
    // Defaults
    next_state     = current_state;
    busy           = 1'b0;
    done           = 1'b0;
    load_bias      = 1'b0;

    in_height      = CONV_IN_H[layer_count];
    in_width       = CONV_IN_W[layer_count];
    in_ch          = CONV_IN_C[layer_count];
    out_ch         = CONV_OUT_C[layer_count];
    kernel_h       = CONV_KH[layer_count];
    kernel_w       = CONV_KW[layer_count];
    stride_h       = CONV_STR_H[layer_count];
    stride_w       = CONV_STR_W[layer_count];

    bypass_maxpool = 1'b0;

    layer_idx      = layer_count;

    case (current_state)
      S_IDLE: begin
        if (start)
          next_state = S_LOAD_BIAS_1;
      end

      S_LOAD_BIAS_1: begin
        // Flash load_bias to bias ROM
        busy = 1'b1;
        load_bias = 1'b1;
        next_state = S_LOAB_BIAS_2;
      end

      S_LOAD_BIAS_2: begin
      	// Bias ROM now responded to load, values loaded into STA
      	// after this cycle
      	busy = 1'b1;
      	load_bias = 1'b0;
      	next_state = S_RUN;
      end

      S_RUN: begin
        // STA controller should run using above parameters
        busy = 1'b1;
        // Here you would poll a 'conv_done' or 'layer_done' from STA
        // For now, we just assume one cycle
        next_state = (layer_count == NUM_LAYERS-1) ? S_DONE : S_LOAD;
      end

      S_STALL: begin

      end

      S_WAIT: begin
      	next_state = (sta_idle) ? S_DONE : S_WAIT;
      end

      S_DONE: begin
        done       = 1'b1;
        next_state = S_IDLE;
      end
    endcase
  end

endmodule

// 1. Load bias
// 2. Start sending inputs to STA, let STA know when inputs are valid
// 2a. Drive STA inputs properly
// 3. Handle STA outputs
// 4. Detect when calculation is complete
	// Each element of STA controller will need idle signal
	// When all elements idle after calculation, it is finished