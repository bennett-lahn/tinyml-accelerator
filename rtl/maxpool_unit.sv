`include "sys_types.svh"

// Parameterized MaxPool2D unit: computes max over a FILTER_H×FILTER_W window of int8 inputs.
// Pooling does not change quantization parameters, so no requantize needed.
module maxpool_unit #(
  parameter int FILTER_H = 2,           // pooling window height
  parameter int FILTER_W = 2            // pooling window width
)(
  input  int8_t in [FILTER_H*FILTER_W], // flattened window inputs
  output int8_t out                     // maximum of all inputs
);

  // Total number of elements in the pooling window
  localparam int N = FILTER_H * FILTER_W;

  // Temporary value to hold running maximum
  int8_t tmp_max;

  always_comb begin
    // Initialize with the first element
    tmp_max = in[0];

    // Iterate over all remaining elements
    for (int i = 1; i < N; i++) begin
      if (in[i] > tmp_max)
        tmp_max = in[i];
    end

    out = tmp_max;
  end

endmodule
