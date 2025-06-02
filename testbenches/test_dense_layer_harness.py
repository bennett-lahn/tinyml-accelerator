import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np

async def load_memories(dut, input_vector, weight_matrix):
    """Load test data into the memory modules (except bias ROM which is file-initialized)"""
    
    # Load tensor RAM (input vector)
    for i, val in enumerate(input_vector):
        dut.tensor_ram_we.value = 1
        dut.tensor_ram_init_addr.value = i
        dut.tensor_ram_init_data.value = int(val)
        await RisingEdge(dut.clk)
    
    dut.tensor_ram_we.value = 0
    await RisingEdge(dut.clk)
    
    # Load weight ROM (weight matrix in row-major order)
    weight_idx = 0
    for row in weight_matrix:
        for val in row:
            dut.weight_rom_we.value = 1
            dut.weight_rom_init_addr.value = weight_idx
            dut.weight_rom_init_data.value = int(val)
            weight_idx += 1
            await RisingEdge(dut.clk)
    
    dut.weight_rom_we.value = 0
    await RisingEdge(dut.clk)

@cocotb.test()
async def test_dense_layer_harness_basic(dut):
    """Test basic dense layer computation with real memory modules"""
    
    # Enable VCD dumping
    dut._log.info("Starting VCD dump")
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    
    # Initialize memory control signals
    dut.tensor_ram_we.value = 0
    dut.weight_rom_we.value = 0
    
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)
    
    # Test data with known bias values from hex file
    input_vector = [2, 3, 1]
    weight_matrix = [[1, 2, 3],   # weights for output 0
                     [4, 5, 6]]   # weights for output 1
    # Bias values from hex file: bias[0]=10, bias[1]=20
    bias_vector = [10, 20]
    
    # Load test data into memories (no bias loading since fc_bias_rom uses file init)
    await load_memories(dut, input_vector, weight_matrix)
    
    # Configure dense layer
    dut.input_size.value = len(input_vector)
    dut.output_size.value = len(weight_matrix)  # 2 outputs
    dut.input_valid.value = 1
    
    # Start computation
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for outputs and collect results
    outputs = {}
    cycle_count = 0
    max_cycles = 50
    
    while not dut.computation_complete.value and cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Debug: Monitor input_idx changes within COMPUTE_MAC state
        if hasattr(dut.dut, 'current_state') and hasattr(dut.dut, 'input_idx'):
            current_state = int(dut.dut.current_state.value)
            if current_state == 3:  # COMPUTE_MAC state (was 2, now 3 after adding LOAD_DATA)
                input_idx = int(dut.dut.input_idx.value)
                input_size = int(dut.input_size.value)
                # Also monitor MAC inputs
                if hasattr(dut, 'tensor_ram_dout') and hasattr(dut, 'weight_rom_dout'):
                    tensor_val = int(dut.tensor_ram_dout.value)
                    weight_val = int(dut.weight_rom_dout.value)
                    # Convert to signed int8 for display
                    tensor_signed = tensor_val if tensor_val < 128 else tensor_val - 256
                    weight_signed = weight_val if weight_val < 128 else weight_val - 256
                    print(f"*** COMPUTE_MAC: input_idx={input_idx}, tensor={tensor_signed}, weight={weight_signed}, product={tensor_signed*weight_signed} ***")
                else:
                    print(f"*** COMPUTE_MAC: input_idx={input_idx}, input_size={input_size}, condition={input_idx < input_size - 1} ***")
        
        # Debug: Print state transitions and key indices
        if hasattr(dut.dut, 'current_state') and hasattr(dut.dut, 'next_state'):
            current_state = int(dut.dut.current_state.value)
            next_state = int(dut.dut.next_state.value)
            if current_state != next_state:
                state_names = ['IDLE', 'LOAD_BIAS', 'LOAD_DATA', 'COMPUTE_MAC', 'OUTPUT_READY', 'COMPLETE']
                curr_name = state_names[current_state] if current_state < len(state_names) else str(current_state)
                next_name = state_names[next_state] if next_state < len(state_names) else str(next_state)
                # Also print key indices during state transitions
                if hasattr(dut.dut, 'input_idx') and hasattr(dut.dut, 'output_idx'):
                    input_idx = int(dut.dut.input_idx.value)
                    output_idx = int(dut.dut.output_idx.value)
                    input_size = int(dut.input_size.value)
                    output_size = int(dut.output_size.value)
                    print(f"*** State transition: {curr_name} -> {next_name} (input_idx={input_idx}, output_idx={output_idx}, input_size={input_size}, output_size={output_size}) ***")
                else:
                    print(f"*** State transition: {curr_name} -> {next_name} ***")
        
        # Debug: Print bias ROM reads
        if dut.bias_rom_re.value:
            bias_addr = int(dut.bias_rom_addr.value)
            bias_data = int(dut.bias_rom_dout.value)
            # Also check output_idx in dense layer compute
            if hasattr(dut.dut, 'output_idx'):
                output_idx = int(dut.dut.output_idx.value)
                print(f"*** Bias ROM read: addr={bias_addr}, data={bias_data}, output_idx={output_idx} ***")
            else:
                print(f"*** Bias ROM read: addr={bias_addr}, data={bias_data} ***")
        
        # Debug: Print MAC unit bias loading
        if hasattr(dut.dut, 'mac_load_bias') and dut.dut.mac_load_bias.value:
            if hasattr(dut.dut, 'mac_bias_in'):
                bias_in = int(dut.dut.mac_bias_in.value)
                # Also get current output channel being computed
                if hasattr(dut.dut, 'current_output_channel'):
                    out_ch = int(dut.dut.current_output_channel.value)
                    print(f"*** MAC load bias for output {out_ch}: bias_in={bias_in} ***")
                else:
                    print(f"*** MAC load bias: bias_in={bias_in} ***")
            else:
                print(f"*** MAC load bias signal asserted ***")
        
        if dut.output_ready.value:
            channel = int(dut.output_channel.value)
            data = int(dut.output_data.value)
            outputs[channel] = data
            print(f"*** Received output for channel {channel}: {data} ***")
    
    print(f"*** Computation completed at cycle {cycle_count} ***")
    
    # Calculate expected outputs with bias values from hex file
    expected_outputs = {}
    for i in range(len(weight_matrix)):
        dot_product = sum(input_vector[j] * weight_matrix[i][j] for j in range(len(input_vector)))
        expected_outputs[i] = dot_product + bias_vector[i]
    
    # Verify outputs
    for i in range(len(weight_matrix)):
        actual_output = outputs.get(i, None)
        expected_output = expected_outputs[i]
        print(f"Output {i}: Expected {expected_output}, Got {actual_output}")
        assert actual_output is not None, f"Missing output for channel {i}"
        assert actual_output == expected_output, f"Output {i} mismatch: expected {expected_output}, got {actual_output}"
    
    print("✅ Basic harness test passed!")

@cocotb.test()
async def test_dense_layer_harness_memory_timing(dut):
    """Test that memory timing works correctly"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    
    # Initialize memory control signals
    dut.tensor_ram_we.value = 0
    dut.weight_rom_we.value = 0
    
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)
    
    # Simple test data with bias values from hex file
    input_vector = [1, 2]
    weight_matrix = [[1, 2],   # weights for output 0: 1*1 + 2*2 + 10 = 15
                     [3, 4]]   # weights for output 1: 1*3 + 2*4 + 20 = 31
    # Bias values from hex file: bias[0]=10, bias[1]=20
    bias_vector = [10, 20]
    
    # Load test data into memories
    await load_memories(dut, input_vector, weight_matrix)
    
    # Configure dense layer
    dut.input_size.value = len(input_vector)
    dut.output_size.value = len(weight_matrix)
    dut.input_valid.value = 1
    
    # Start computation
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for outputs and collect results
    outputs = {}
    cycle_count = 0
    max_cycles = 30
    
    while not dut.computation_complete.value and cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        if dut.output_ready.value:
            channel = int(dut.output_channel.value)
            data = int(dut.output_data.value)
            outputs[channel] = data
            print(f"Output {channel}: {data}")
    
    # Verify expected outputs with bias values
    assert outputs[0] == 15, f"Output 0: expected 15, got {outputs[0]}"
    assert outputs[1] == 31, f"Output 1: expected 31, got {outputs[1]}"
    
    print("✅ Memory timing test passed!") 