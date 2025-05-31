import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_dense_layer_basic_functionality(dut):
    """Test basic dense layer computation with small matrices"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 0
    dut.output_size.value = 0
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 0
    dut.weight_rom_dout.value = 0
    dut.bias_rom_dout.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test configuration: 4 inputs, 3 outputs
    input_size = 4
    output_size = 3
    
    # Test data
    # Input vector: [1, 2, 3, 4] (signed 8-bit)
    input_vector = np.array([1, 2, 3, 4], dtype=np.int8)
    
    # Weight matrix (4x3): each column is weights for one output neuron
    # Output 0: [1, 1, 1, 1] -> dot product = 1*1 + 2*1 + 3*1 + 4*1 = 10
    # Output 1: [2, 1, 0, -1] -> dot product = 1*2 + 2*1 + 3*0 + 4*(-1) = 0  
    # Output 2: [0, 1, 2, 1] -> dot product = 1*0 + 2*1 + 3*2 + 4*1 = 12
    weight_matrix = np.array([
        [1, 2, 0],   # weights for input 0
        [1, 1, 1],   # weights for input 1  
        [1, 0, 2],   # weights for input 2
        [1, -1, 1]   # weights for input 3
    ], dtype=np.int8)
    
    # Bias vector: [5, -3, 7]
    bias_vector = np.array([5, -3, 7], dtype=np.int32)
    
    # Expected outputs: [10+5, 0+(-3), 12+7] = [15, -3, 19]
    expected_outputs = np.array([15, -3, 19], dtype=np.int32)
    
    dut.input_size.value = input_size
    dut.output_size.value = output_size
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for state machine to start
    await RisingEdge(dut.clk)
    
    # Monitor state transitions and provide memory responses
    state_history = []
    cycle_count = 0
    max_cycles = 100
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Record current state (assuming we can read internal signals)
        current_state = int(dut.current_state.value) if hasattr(dut, 'current_state') else -1
        state_history.append(current_state)
        
        # Respond to memory requests
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = int(input_vector[addr])
            else:
                dut.tensor_ram_dout.value = 0
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            # Address calculation: input_idx * output_size + output_idx
            if addr < weight_matrix.size:
                input_idx = addr // output_size
                output_idx = addr % output_size
                if input_idx < weight_matrix.shape[0] and output_idx < weight_matrix.shape[1]:
                    dut.weight_rom_dout.value = int(weight_matrix[input_idx, output_idx])
                else:
                    dut.weight_rom_dout.value = 0
            else:
                dut.weight_rom_dout.value = 0
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                dut.bias_rom_dout.value = int(bias_vector[addr])
            else:
                dut.bias_rom_dout.value = 0
        
        # Check if computation is complete
        if int(dut.computation_complete.value) == 1:
            print(f"Computation completed at cycle {cycle_count}")
            break
    
    # Verify outputs
    assert int(dut.computation_complete.value) == 1, "Computation should be complete"
    assert int(dut.output_valid.value) == 1, "Output should be valid"
    
    # Check output values
    for i in range(output_size):
        actual_output = int(dut.output_vector[i].value)
        expected_output = expected_outputs[i]
        print(f"Output {i}: Expected {expected_output}, Got {actual_output}")
        assert actual_output == expected_output, f"Output {i} mismatch: expected {expected_output}, got {actual_output}"
    
    # Verify configuration echo
    assert int(dut.current_input_size.value) == input_size, "Input size should be echoed correctly"
    assert int(dut.current_output_size.value) == output_size, "Output size should be echoed correctly"
    
    print("✅ Basic dense layer computation test passed!")

@cocotb.test()
async def test_dense_layer_state_machine(dut):
    """Test the state machine transitions"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 2
    dut.output_size.value = 2
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 1
    dut.weight_rom_dout.value = 1
    dut.bias_rom_dout.value = 100
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Should be in IDLE state (0)
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 0, "Should start in IDLE state"
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Should transition to LOAD_BIAS state (1)
    await RisingEdge(dut.clk)
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 1, "Should be in LOAD_BIAS state"
    
    # Wait for bias loading to complete (2 bias values)
    bias_cycles = 0
    while bias_cycles < 10:  # Safety limit
        await RisingEdge(dut.clk)
        bias_cycles += 1
        if hasattr(dut, 'current_state') and int(dut.current_state.value) == 2:  # COMPUTE_MAC
            break
    
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 2, "Should transition to COMPUTE_MAC state"
    
    # Wait for computation to complete
    compute_cycles = 0
    while compute_cycles < 20:  # Safety limit
        await RisingEdge(dut.clk)
        compute_cycles += 1
        if int(dut.computation_complete.value) == 1:
            break
    
    # Should be in COMPLETE state (3)
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 3, "Should be in COMPLETE state"
    
    print("✅ State machine test passed!")

@cocotb.test()
async def test_dense_layer_memory_addressing(dut):
    """Test memory address generation"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 3
    dut.output_size.value = 2
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 1
    dut.weight_rom_dout.value = 1
    dut.bias_rom_dout.value = 100
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Track memory accesses
    tensor_addresses = []
    weight_addresses = []
    bias_addresses = []
    
    cycle_count = 0
    while cycle_count < 50:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Record memory accesses
        if int(dut.tensor_ram_re.value) == 1:
            tensor_addresses.append(int(dut.tensor_ram_addr.value))
            
        if int(dut.weight_rom_re.value) == 1:
            weight_addresses.append(int(dut.weight_rom_addr.value))
            
        if int(dut.bias_rom_re.value) == 1:
            bias_addresses.append(int(dut.bias_rom_addr.value))
        
        if int(dut.computation_complete.value) == 1:
            break
    
    # Verify bias addresses (should be 0, 1 for 2 outputs)
    expected_bias_addrs = [0, 1]
    assert bias_addresses == expected_bias_addrs, f"Bias addresses: expected {expected_bias_addrs}, got {bias_addresses}"
    
    # Verify tensor RAM addresses (should access inputs 0, 1, 2 multiple times)
    expected_tensor_addrs = [0, 1, 2] * 2  # 3 inputs × 2 outputs
    assert set(tensor_addresses) == set([0, 1, 2]), f"Tensor addresses should include all inputs"
    
    # Verify weight ROM addresses
    # For 3 inputs × 2 outputs, addresses should be:
    # input_idx * output_size + output_idx
    # (0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3, 2*2+0=4, 2*2+1=5)
    expected_weight_addrs = [0, 1, 2, 3, 4, 5]
    assert set(weight_addresses) == set(expected_weight_addrs), f"Weight addresses: expected {expected_weight_addrs}, got unique {list(set(weight_addresses))}"
    
    print("✅ Memory addressing test passed!")

@cocotb.test()
async def test_dense_layer_edge_cases(dut):
    """Test edge cases like single input/output"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test 1: Single input, single output
    dut.input_size.value = 1
    dut.output_size.value = 1
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    # Test data: input=5, weight=3, bias=2 -> output=5*3+2=17
    dut.tensor_ram_dout.value = 5
    dut.weight_rom_dout.value = 3
    dut.bias_rom_dout.value = 2
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for completion
    cycle_count = 0
    while cycle_count < 20:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if int(dut.computation_complete.value) == 1:
            break
    
    assert int(dut.computation_complete.value) == 1, "Single input/output computation should complete"
    expected_output = 5 * 3 + 2  # 17
    actual_output = int(dut.output_vector[0].value)
    assert actual_output == expected_output, f"Single I/O: expected {expected_output}, got {actual_output}"
    
    print("✅ Edge cases test passed!")

if __name__ == "__main__":
    print("Dense layer compute testbench") 