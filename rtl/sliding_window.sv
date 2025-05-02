`include "./sys_types.svh"
// typedef logic signed [7:0]   int8_t;
// typedef logic signed [15:0] int16_t;
// typedef logic signed [31:0] int32_t;
//takes in a pixel per clock cycle and builds the 4x4 output
//new window every clock cycle!
module sliding_window #(
    parameter IMG_W = 96,
    parameter IMG_H = 96
) (
    input logic clk
    ,input logic reset
    ,input logic valid_in
    ,input int8_t pixel_in
    ,output logic valid_out
    ,output int8_t A0 [0:3]
    ,output int8_t A1 [0:3]
    ,output int8_t A2 [0:3]
    ,output int8_t A3 [0:3]
);

    logic [$clog2(IMG_W)-1:0] col_counter;
    logic [$clog2(IMG_H)-1:0] row_counter;

    int8_t line_buff0 [0:IMG_W-1], line_buff1 [0:IMG_W-1], line_buff2 [0:IMG_W-1];
    int8_t sr0 [0:3], sr1 [0:3], sr2 [0:3], sr3 [0:3];

    // delayed srs 
    int8_t sr1_c1 [0:3];
    int8_t sr2_c1 [0:3], sr2_c2 [0:3];
    int8_t sr3_c1 [0:3], sr3_c2 [0:3], sr3_c3[0:3];

    logic valid_d1, valid_d2, valid_d3;
    logic valid0;

    int8_t d1, d2, d3;

    assign valid0 = valid_in && (col_counter >= 3) && (row_counter >= 3);

    always_ff @(posedge clk) begin
        if(reset)begin
            col_counter <= 0;
            row_counter <= 0;

            sr1_c1 <= '{default:0};
            sr2_c1 <= '{default:0};
            sr2_c2 <= '{default:0};
            sr3_c1 <= '{default:0};
            sr3_c2 <= '{default:0};
            sr3_c3 <= '{default:0};
            valid_d1 <= 0;
            valid_d2 <= 0;
            valid_d3 <= 0;
        end
        else if(valid_in) begin
            if(col_counter == IMG_W-1) begin
                col_counter <= 0;
                if(row_counter == IMG_H-1) begin
                    row_counter <= 0;
                end
                else begin
                    row_counter <= row_counter + 1;
                end
            end
            else begin
                col_counter <= col_counter + 1;
            end
            line_buff2[col_counter] <= line_buff1[col_counter];
            line_buff1[col_counter] <= line_buff0[col_counter];
            line_buff0[col_counter] <= pixel_in;

            d1 <= line_buff0[col_counter];
            d2 <= line_buff1[col_counter];
            d3 <= line_buff2[col_counter];

            sr0 <= { sr0[0:2], pixel_in};
            sr1 <= { sr1[0:2], d1};
            sr2 <= { sr2[0:2], d2};
            sr3 <= { sr3[0:2], d3};


            // pipelining to add delays 
            sr1_c1 <= sr1;
            sr2_c1 <= sr2; 
            sr2_c2 <= sr2_c1;
            sr3_c1 <= sr3;
            sr3_c1 <= sr3_c2;
            sr3_c2 <= sr3_c1;
            sr3_c3 <= sr3_c2;

            valid_d1 <= valid0;
            valid_d2 <= valid_d1;
            valid_d3 <= valid_d2;


        end
    end

    always_comb begin
        A0 = sr0;
        A1 = sr1_c1;
        A2 = sr2_c2;
        A3 = sr3_c3;

        valid_out = valid_d3;
    end






endmodule 