module pointer #(
    parameter DEPTH = 16384
) (
    input logic clk
    ,input logic reset
    ,input logic reset_ptr
    //control signal when we move on to the next input channel
    ,input logic incr_ptr
    ,input logic [$clog2(DEPTH)-1:0] rst_ptr_val
    ,output logic [$clog2(DEPTH)-1:0] ptr

);

    always_ff @(posedge clk) begin
        if (reset) begin
            ptr <= 0;
        end
        else if (reset_ptr) begin
            ptr <= rst_ptr_val;
        end
        else if (incr_ptr) begin
            ptr <= ptr + 1;
        end
    end

endmodule 
