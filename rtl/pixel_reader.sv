module pixel_reader #(
    parameter DEPTH = 96*96
) (
    input logic clk
    ,input logic reset
    ,input logic start
    ,input logic incr
    
    ,output logic valid_out
    ,output logic [$clog2(DEPTH)-1:0] pixel_ptr
);

    // localparam IMG_SIZE = IMG_W * IMG_H;
    localparam ADDR_WIDTH = $clog2(DEPTH);

    typedef enum logic {Idle, Stream} state_t;
    state_t state, next_state;

    logic [ADDR_WIDTH-1:0] ptr;

    always_ff @(posedge clk) begin
        if(reset) begin
            state <= Idle;
            ptr <= 0;
        end
        else begin
            state <= next_state;
            case (state)
                Idle: begin
                    if(start) begin
                        ptr <= 0;
                    end
                end
                Stream: begin
                    if(ptr != ADDR_WIDTH'(DEPTH-1)) begin
                        ptr <= ptr + 1;
                    end
                end
            endcase
        end
    end


    always_comb begin
        valid_out = 0;
        pixel_ptr = ptr;
        next_state = state;
        case (state)
            Idle: begin
                if(start) begin
                    next_state = Stream;
                    valid_out = 1;
                end
            end
            Stream: begin
                valid_out = 1;
                if(ptr == ADDR_WIDTH'(DEPTH-1)) begin
                    next_state = Idle;
                end
            end
        endcase
    end



endmodule 
