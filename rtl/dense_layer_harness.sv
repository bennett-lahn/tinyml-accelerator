`include "sys_types.svh"

module dense_layer_harness (
    input logic clk,
    input logic reset,
    
    // Control signals
    input logic start_compute,
    input logic input_valid,
    
    // Runtime configuration inputs
    input logic [$clog2(256+1)-1:0] input_size,    // Actual input size (up to 256)
    input logic [$clog2(64+1)-1:0] output_size,    // Actual output size (up to 64)
    
    // Memory initialization ports (for loading test data)
    input logic tensor_ram_we,
    input logic [$clog2(256)-1:0] tensor_ram_init_addr,
    input logic [7:0] tensor_ram_init_data,
    
    input logic weight_rom_we,
    input logic [$clog2(256*64)-1:0] weight_rom_init_addr,
    input logic [7:0] weight_rom_init_data,
    
    // Single output (one result at a time)
    output logic [31:0] output_data,
    output logic [$clog2(64)-1:0] output_channel,  // Which output channel this result is for
    output logic output_ready,                     // Indicates output_data is valid for current channel
    output logic computation_complete,             // All outputs computed
    
    // Status outputs
    output logic [$clog2(256+1)-1:0] current_input_size,
    output logic [$clog2(64+1)-1:0] current_output_size
);

    // Internal signals for memory interfaces
    logic [$clog2(256)-1:0] tensor_ram_addr;
    logic tensor_ram_re;
    logic [7:0] tensor_ram_dout;
    
    logic [$clog2(256*64)-1:0] weight_rom_addr;
    logic weight_rom_re;
    logic [7:0] weight_rom_dout;
    
    logic [$clog2(64)-1:0] bias_rom_addr;
    logic bias_rom_re;
    logic [31:0] bias_rom_dout;

    // Tensor RAM instantiation (simple dual-port RAM)
    logic [$clog2(256)-1:0] tensor_ram_addr_mux;
    logic tensor_ram_re_mux;
    logic tensor_ram_we_mux;
    logic [7:0] tensor_ram_din_mux;
    
    // Mux between initialization and compute access
    always_comb begin
        if (tensor_ram_we) begin
            // Initialization mode
            tensor_ram_addr_mux = tensor_ram_init_addr;
            tensor_ram_we_mux = tensor_ram_we;
            tensor_ram_re_mux = 1'b0;
            tensor_ram_din_mux = tensor_ram_init_data;
        end else begin
            // Compute mode
            tensor_ram_addr_mux = tensor_ram_addr;
            tensor_ram_we_mux = 1'b0;
            tensor_ram_re_mux = tensor_ram_re;
            tensor_ram_din_mux = 8'b0;
        end
    end

    // Simple RAM for tensor data
    logic [7:0] tensor_ram [0:255];
    always_ff @(posedge clk) begin
        if (tensor_ram_we_mux) begin
            tensor_ram[tensor_ram_addr_mux] <= tensor_ram_din_mux;
        end
        if (tensor_ram_re_mux) begin
            tensor_ram_dout <= tensor_ram[tensor_ram_addr_mux];
        end
    end

    // Weight ROM instantiation
    logic [$clog2(256*64)-1:0] weight_rom_addr_mux;
    logic weight_rom_re_mux;
    logic weight_rom_we_mux;
    logic [7:0] weight_rom_din_mux;
    
    // Mux between initialization and compute access
    always_comb begin
        if (weight_rom_we) begin
            // Initialization mode
            weight_rom_addr_mux = weight_rom_init_addr;
            weight_rom_we_mux = weight_rom_we;
            weight_rom_re_mux = 1'b0;
            weight_rom_din_mux = weight_rom_init_data;
        end else begin
            // Compute mode
            weight_rom_addr_mux = weight_rom_addr;
            weight_rom_we_mux = 1'b0;
            weight_rom_re_mux = weight_rom_re;
            weight_rom_din_mux = 8'b0;
        end
    end

    // Simple RAM for weight data (acting as ROM)
    logic [7:0] weight_rom [0:256*64-1];
    always_ff @(posedge clk) begin
        if (weight_rom_we_mux) begin
            weight_rom[weight_rom_addr_mux] <= weight_rom_din_mux;
        end
        if (weight_rom_re_mux) begin
            weight_rom_dout <= weight_rom[weight_rom_addr_mux];
        end
    end

    // FC Bias ROM instantiation using existing fc_bias_rom module
    fc_bias_rom #(
        .WIDTH(32),
        .DEPTH(64),         // Only need 64 entries for our dense layer
        .INIT_FILE("test_bias_values.hex"),     // Use test bias values
        .FC1_SIZE(64),      // Set to our max output size
        .FC2_SIZE(0)        // Not used, but keep parameter
    ) bias_rom_inst (
        .clk(clk),
        .read_enable(bias_rom_re),
        .fc_layer_select(1'b0),    // Always use FC1 layer (first layer)
        .addr(bias_rom_addr[$clog2(64)-1:0]),  // Limit to 6 bits for 64 addresses
        .bias_out(bias_rom_dout)
    );

    // Dense layer compute instantiation
    dense_layer_compute dut (
        .clk(clk),
        .reset(reset),
        .start_compute(start_compute),
        .input_valid(input_valid),
        .input_size(input_size),
        .output_size(output_size),
        .tensor_ram_addr(tensor_ram_addr),
        .tensor_ram_re(tensor_ram_re),
        .tensor_ram_dout(tensor_ram_dout),
        .weight_rom_addr(weight_rom_addr),
        .weight_rom_re(weight_rom_re),
        .weight_rom_dout(weight_rom_dout),
        .bias_rom_addr(bias_rom_addr),
        .bias_rom_re(bias_rom_re),
        .bias_rom_dout(bias_rom_dout),
        .output_data(output_data),
        .output_channel(output_channel),
        .output_ready(output_ready),
        .computation_complete(computation_complete),
        .current_input_size(current_input_size),
        .current_output_size(current_output_size)
    );

endmodule 
