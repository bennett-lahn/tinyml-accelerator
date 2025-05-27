`include "sys_types.svh"

module TPU_Datapath #(
    parameter IMG_W = 32
    ,parameter IMG_H = 32
    ,parameter MAX_N = 64 // Default MAX_N from sta_controller
    ,parameter MAX_NUM_CH = 64
    ,parameter STA_CTRL_N_BITS = $clog2(MAX_N + 1) // Should be 9 bits for MAX_N=512
    ,parameter SA_N = 4  // SA_N from sta_controller
    ,parameter SA_VECTOR_WIDTH = 4 // SA_VECTOR_WIDTH from sta_controller
    ,parameter TOTAL_PES_STA = SA_N * SA_N // Should be 16
    ,parameter NUM_LAYERS = 6 // Number of layers in the model
) (
    input logic clk
    ,input logic reset
    ,input logic read_weights
    ,input logic read_inputs
    ,input logic start
    ,input logic done
);

// STA Controller signals
logic stall;

logic                          load_bias;                    // Single load_bias control signal
int32_t                        bias_value;                  // Single bias value to be used for all PEs

logic                          idle;                          // High if STA complex is idle
logic                          array_out_valid;               // High if corresponding val/row/col is valid
logic [127:0]                  array_val_out;                 // Value out of max pool unit
logic [$clog2(MAX_N)-1:0]    array_row_out;                 // Row for corresponding value
logic [$clog2(MAX_N)-1:0]    array_col_out;                 // Column for corresponding value

// logic                          start;       // Pulse to begin sequence
logic					       sta_idle;    // STA signals completion of current tile
// logic                          done;        // Signals completion of current tile computation

// Current layer and channel index
logic [$clog2(NUM_LAYERS)-1:0] layer_idx; // Index of current layer
logic [$clog2(MAX_NUM_CH)-1:0] chnnl_idx; // Index of current output channel / filter

// Drive STA controller and memory conv parameters
logic                          reset_sta;
logic [15:0]                   mat_size;
logic                          start_compute;
logic [$clog2(MAX_N)-1:0]      controller_pos_row;
logic [$clog2(MAX_N)-1:0]      controller_pos_col;

// Current layer filter dimensions
logic [$clog2(MAX_NUM_CH+1)-1:0] num_filters;      // Number of output channels (filters) for current layer
logic [$clog2(MAX_NUM_CH+1)-1:0] num_input_channels; // Number of input channels for current layer

// Drive STA controller pool parameters
logic 				           bypass_maxpool;

    // layer_controller layer_controller (
    //     .clk(clk)
    //     ,.reset(reset)

    //     // Inputs to layer_controller
    //     ,.start(start)
    //     ,.stall(stall)
    //     ,.sta_idle(sta_idle)
    //     ,.done(done)
    //     ,.layer_idx(layer_idx)
    //     ,.chnnl_idx(chnnl_idx)
    //     ,.reset_sta(reset_sta)

    //     // Outputs from layer_controller
    //     ,.mat_size(mat_size)
    //     ,.load_bias(load_bias)
    //     ,.start_compute(start_compute)
    //     ,.controller_pos_row(controller_pos_row)
    //     ,.controller_pos_col(controler_pos_col)
    //     ,.num_filters(num_filters)
    //     ,.num_input_channels(num_input_channels)
    // );

    sta_controller sta_controller (
        .clk(clk)
        ,.reset(reset)
        ,.reset_sta('0)
        // Inputs to sta_controller
        ,.stall('0)
        ,.bypass_maxpool('0)
        ,.controller_pos_row('0)
        ,.controller_pos_col('0)
        ,.done(done)
        ,.layer_idx('0)
        ,.A0(a0_IN_KERNEL)
        ,.A1(a1_IN_KERNEL)
        ,.A2(a2_IN_KERNEL)
        ,.A3(a3_IN_KERNEL)
        ,.B0(weight_C0)
        ,.B1(weight_C1)
        ,.B2(weight_C2)
        ,.B3(weight_C3)
        ,.load_bias('0)
        ,.bias_value('0) // Assuming bias_value is a 32-bit value, adjust as needed

        // Outputs from sta_controller
        ,.idle(idle)
        ,.array_out_valid(array_out_valid)
        ,.array_val_out(array_val_out)
        ,.array_row_out(array_row_out)
        ,.array_col_out(array_col_out)
    );




    //======================================================================================================

    //Tensor RAM A, and B will be used to store and write , vice versa. never read and write to the same mememory. 
    logic [31:0] tensor_ram_A_dout0;
    logic [31:0] tensor_ram_A_dout1;
    logic [31:0] tensor_ram_A_dout2;
    logic [31:0] tensor_ram_A_dout3;
    logic [127:0] tensor_ram_A_din;
    logic ram_A_we;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_A_addr_w;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_A_addr_r;
    // Assuming IMG_W and IMG_H are defined in the module scope, e.g., as parameters or localparams
    // Instantiate tensor_ram for A
    tensor_ram 
    #(
        .D_WIDTH(128) // 4 8-bit pixels per read/write
        , .DEPTH(IMG_W * IMG_H) // Assuming IMG_W and IMG_H are defined in the module scope
        , .INIT_FILE("../rtl/image_data.hex") // No initialization file specified
    )
    RAM_A
    (
        .clk(clk)
        ,.we(ram_A_we)
        ,.addr_w(ram_A_addr_w)
        ,.din(tensor_ram_A_din)
        ,.addr_r(ram_A_addr_r)
        ,.dout0(tensor_ram_A_dout0)
        ,.dout1(tensor_ram_A_dout1)
        ,.dout2(tensor_ram_A_dout2)
        ,.dout3(tensor_ram_A_dout3)
    );


    //======================================================================================================
    //Tensor RAM B, and A will be used to store and write , vice versa. never read and write to the same mememory.
    logic [31:0] tensor_ram_B_dout0;
    logic [31:0] tensor_ram_B_dout1;
    logic [31:0] tensor_ram_B_dout2;
    logic [31:0] tensor_ram_B_dout3;
    logic [127:0] tensor_ram_B_din;
    logic ram_B_we;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_B_addr_w;
    logic [($clog2(IMG_W * IMG_H))-1:0] ram_B_addr_r;

    
    
    // Instantiate tensor_ram for B
    tensor_ram
    #(
        .D_WIDTH(128) // 4 8-bit pixels per read/write
        , .DEPTH(IMG_W * IMG_H) // Assuming IMG_W and IMG_H are defined in the module scope
        , .INIT_FILE("") // No initialization file specified
    )
    RAM_B
    (
        .clk(clk)
        ,.we(ram_B_we)
        ,.addr_w(ram_B_addr_w)
        ,.din(tensor_ram_B_din)
        ,.addr_r(ram_B_addr_r)
        ,.dout0(tensor_ram_B_dout0)
        ,.dout1(tensor_ram_B_dout1)
        ,.dout2(tensor_ram_B_dout2)
        ,.dout3(tensor_ram_B_dout3)
    );



    //======================================================================================================
    logic  incr_ptr_A; // Pointer for reading from tensor_ram A
    pointer 
    #(
        .DEPTH(IMG_W * IMG_H) // Assuming IMG_W and IMG_H are defined in the module scope
    )
    pixel_pointer_A
    (
        .clk(clk)
        ,.reset(reset)
        ,.incr_ptr(incr_ptr_A) // Assuming this is the control signal to increment the pointer
        ,.ptr(ram_A_addr_r) // Output pointer for reading from tensor_ram A
    );

    logic incr_ptr_B; // Pointer for reading from tensor_ram B
    pointer
    #(
        .DEPTH(IMG_W * IMG_H) // Assuming IMG_W and IMG_H are defined in the module scope
    )
    pixel_pointer_B
    (
        .clk(clk)
        ,.reset(reset)
        ,.incr_ptr(incr_ptr_B) // Assuming this is the control signal to increment the pointer
        ,.ptr(ram_B_addr_r) // Output pointer for reading from tensor_ram B
    );




    //======================================================================================================
    // SLIDING WINDOW for input images
    logic start_sliding_window; // Control signal to start the sliding window operation
    logic valid_in_sliding_window; // Valid signal for the sliding window input
    int8_t a0_IN_KERNEL [3:0]; // Assuming 4 input channels for the sliding window
    int8_t a1_IN_KERNEL [3:0]; // Assuming 4 input channels for the sliding window
    int8_t a2_IN_KERNEL [3:0]; // Assuming 4 input channels for the sliding window
    int8_t a3_IN_KERNEL [3:0]; // Assuming 4 input channels for the sliding window
    logic valid_IN_KERNEL_A0;
    logic valid_IN_KERNEL_A1;
    logic valid_IN_KERNEL_A2;
    logic valid_IN_KERNEL_A3;

     int32_t IN_KERNAL_WINDOW_A0; //input for the sliding window operation
     int32_t IN_KERNAL_WINDOW_A1; //input for the sliding window operation
     int32_t IN_KERNAL_WINDOW_A2; //input for the sliding window operation
     int32_t IN_KERNAL_WINDOW_A3; //input for the sliding window operation


    logic done_IN_KERNEL; // Signal indicating the sliding window operation is done
    sliding_window IN_KERNELS
    (
        .clk(clk)
        ,.reset(reset)
        ,.start(start_sliding_window)
        ,.valid_in(valid_in_sliding_window)
        ,.A0_in(IN_KERNAL_WINDOW_A0)
        ,.A1_in(IN_KERNAL_WINDOW_A1)
        ,.A2_in(IN_KERNAL_WINDOW_A2)
        ,.A3_in(IN_KERNAL_WINDOW_A3)
        ,.A0(a0_IN_KERNEL)
        ,.A1(a1_IN_KERNEL)
        ,.A2(a2_IN_KERNEL)
        ,.A3(a3_IN_KERNEL)
        ,.valid_A0(valid_IN_KERNEL_A0)
        ,.valid_A1(valid_IN_KERNEL_A1)
        ,.valid_A2(valid_IN_KERNEL_A2)
        ,.valid_A3(valid_IN_KERNEL_A3)
        ,.done(done_IN_KERNEL)
    );



    //======================================================================================================




    //instantiate WEIGHT ROM

    logic WEIGHT_READ_EN;
    logic [($clog2(16384))-1:0] weight_rom_addr; // Assuming 16384 is the depth of the weight ROM
     int32_t weight_rom_dout0; // Output data from the weight ROM
     int32_t weight_rom_dout1; // Output data from the weight ROM
     int32_t weight_rom_dout2; // Output data from the weight ROM
     int32_t weight_rom_dout3; // Output data from the weight ROM

    assign WEIGHT_READ_EN = read_weights; // Control signal to enable reading from weight ROM
    // ROM for weights, assuming 16 8-bit values per read
    weight_rom 
    #(
        .WIDTH(128) // 4 8-bit pixels per read/write
        , .DEPTH(16384) // Assuming IMG_W and IMG_H are defined in the module scope
        , .INIT_FILE("../fakemodel/tflite_conv_kernel_weights.hex") // No initialization file specified
    )
    WEIGHT_ROM
    (
        .clk(clk)
        ,.read_enable(WEIGHT_READ_EN)
        ,.addr(weight_rom_addr)
        ,.data0(weight_rom_dout0)
        ,.data1(weight_rom_dout1)
        ,.data2(weight_rom_dout2)
        ,.data3(weight_rom_dout3)
    );

    logic INCR_WEIGHT_PTR; // Control signal to increment the weight pointer
    pointer weight_pointer
     (
        .clk(clk)
        ,.reset(reset)
        ,.incr_ptr(INCR_WEIGHT_PTR) // Assuming this is the control signal to increment the pointer
        ,.ptr(weight_rom_addr) // Output pointer for reading from weight_rom
     );

    //======================================================================================================

    logic valid_in_sliding_window_weights; // Valid signal for the sliding window input for weights

     int8_t weight_C0 [3:0]; // Assuming 4 input channels for the sliding window weights
     int8_t weight_C1 [3:0]; // Assuming 4 input channels for the sliding window weights
     int8_t weight_C2 [3:0]; // Assuming 4 input channels for the sliding window weights
     int8_t weight_C3 [3:0]; // Assuming 4 input channels for the sliding window weights
    logic valid_C0; // Valid signal for weight_C0
    logic valid_C1; // Valid signal for weight_C1
    logic valid_C2; // Valid signal for weight_C2
    logic valid_C3; // Valid signal for weight_C3
    logic done_WEIGHTS; // Signal indicating the sliding window operation for weights is done
    sliding_window WEIGHTS_IN
    (  
        .clk(clk)
        ,.reset(reset)
        ,.start(start_sliding_window)
        ,.valid_in(valid_in_sliding_window_weights)
        ,.A0_in(weight_rom_dout0)
        ,.A1_in(weight_rom_dout1)
        ,.A2_in(weight_rom_dout2)
        ,.A3_in(weight_rom_dout3)
        ,.A0(weight_C0)
        ,.A1(weight_C1)
        ,.A2(weight_C2)
        ,.A3(weight_C3)
        ,.valid_A0(valid_C0)
        ,.valid_A1(valid_C1)
        ,.valid_A2(valid_C2)
        ,.valid_A3(valid_C3)
        ,.done(done_WEIGHTS)
    );

    //======================================================================================================

    // Control signals for writing and reading if layer index lsb is 0 or 1
    assign ram_A_we = ~layer_idx[0] & array_out_valid; // Example logic to control write enable for tensor_ram A based on layer_idx
    assign ram_B_we = layer_idx[0] & array_out_valid; // Example logic to control write enable for tensor_ram B based on layer_idx
    always_comb begin
        if(read_inputs) begin
        IN_KERNAL_WINDOW_A0 = layer_idx[0] ? tensor_ram_B_dout0 : tensor_ram_A_dout0; // Example logic to select input for sliding window A0
        IN_KERNAL_WINDOW_A1 = layer_idx[0] ? tensor_ram_B_dout1 : tensor_ram_A_dout1; // Example logic to select input for sliding window A1
        IN_KERNAL_WINDOW_A2 = layer_idx[0] ? tensor_ram_B_dout2 : tensor_ram_A_dout2; // Example logic to select input for sliding window A2
        IN_KERNAL_WINDOW_A3 = layer_idx[0] ? tensor_ram_B_dout3 : tensor_ram_A_dout3; // Example logic to select input for sliding window A3
        end else begin
            IN_KERNAL_WINDOW_A0 = 32'b0; // Default value if not reading inputs
            IN_KERNAL_WINDOW_A1 = 32'b0; // Default value if not reading inputs
            IN_KERNAL_WINDOW_A2 = 32'b0; // Default value if not reading inputs
            IN_KERNAL_WINDOW_A3 = 32'b0; // Default value if not reading inputs
        end
    end
     

    //======================================================================================================

    assign start_sliding_window = start; // Control signal to start the sliding window operation

endmodule
