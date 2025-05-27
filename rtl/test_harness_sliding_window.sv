// // `include "./sliding_window.sv"
// // `include "./pixel_reader.sv"

// module test_harness_sliding_window #(
//     parameter IMG_W = 96,
//     parameter IMG_H = 96
// ) (
//     input logic clk
//     ,input logic reset
//     ,input logic start
//     ,output logic valid_A0, valid_A1, valid_A2, valid_A3
//     ,input logic [$clog2(IMG_W*IMG_H)-1:0] addr_w
//     ,output int8_t A0[0:3], A1[0:3], A2[0:3], A3[0:3]
//     ,input logic [31:0] din
//     ,input logic we
// );

//     logic valid;
//     int32_t pixel_in;
//     logic [$clog2(IMG_W*IMG_H)-1:0] pixel_ptr;


//     sliding_window dut (
//         .clk(clk)
//         ,.reset(reset)
//         ,.valid_in(valid)
//         ,.pixels_in_chunk_bus(pixel_in) // Pack 4 copies of pixel_in into 32-bit bus
//         ,.A0(A0)
//         ,.A1(A1)
//         ,.A2(A2)
//         ,.A3(A3)
//         ,.valid_A0(valid_A0)
//         ,.valid_A1(valid_A1)  
//         ,.valid_A2(valid_A2)  
//         ,.valid_A3(valid_A3)  
//     );

//     pixel_reader #(
//         .DEPTH(IMG_W*IMG_H)
//     ) pixel_reader (
//         .clk(clk)
//         ,.reset(reset)
//         ,.start(start)
//         ,.valid_out(valid)
//         ,.pixel_ptr(pixel_ptr)
//     );

//     tensor_ram #(
//         .D_WIDTH(32),
//         .DEPTH(IMG_W*IMG_H),
//         .INIT_FILE("/Users/aadithyamanoj/Documents/GitHub/tinyml-accelerator/src/image_data.hex")
//     ) tensor_ram (
//         .clk(clk)
//         ,.we(we)
//         ,.addr_w(addr_w)
//         ,.din(din)
//         ,.addr_r(pixel_ptr)
//         ,.dout(pixel_in)
//     );



// endmodule 

